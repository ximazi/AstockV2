"""Experiment registry, pre-publish scenario checks and rollback that never rewrites a fill."""
import json
import math
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from . import store

VERSION = "experiment_registry_v1_20260920"
REQUIRED = ("hypothesis", "control", "sample_split", "metric", "max_attempts")
SPLIT_KEYS = ("discover_start", "discover_end", "freeze_date", "oos_start")
DECISIONS = {"PUBLISHED", "VETOED", "ROLLED_BACK"}
# The metric an experiment is *tuned on* stays a discovery metric; return metrics are refused there.
FORBIDDEN_METRICS = ("win_rate", "return", "profit", "sharpe", "pnl")
# The metric a strategy is *accepted on* is economic and frozen at registration, judged once, out of sample.
ECONOMIC_CRITERIA = {"expectancy_net_min": "每笔净期望下限", "max_drawdown_max": "最大回撤上限",
                     "tail_loss_max": "尾部亏损上限", "coverage_min": "覆盖率下限",
                     "n_effective_min": "有效样本（题材×信号日簇）下限"}
ECONOMIC_REQUIRED = ("expectancy_net_min", "n_effective_min")
ECONOMIC_METRICS = {"expectancy_net": "expectancy_net_min", "max_drawdown": "max_drawdown_max",
                    "tail_loss": "tail_loss_max", "coverage": "coverage_min"}


def _finite(value, name, minimum=None, integer=False):
    """A gate that compares against NaN compares against nothing; every number is checked before use."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} 必须是有限数值，不接受空值、布尔值、NaN 或无穷")
    if integer and int(value) != value:
        raise ValueError(f"{name} 必须是整数")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} 不得小于 {minimum}")
    return value


def _date(value, name):
    try:
        return datetime.fromisoformat(str(value)).date().isoformat()
    except (TypeError, ValueError):
        raise ValueError(f"{name} 必须是 ISO 日期") from None
REVIEW_CADENCE = ("2026-09-22", "2026-09-25", "2026-09-28")


def _table(conn):
    conn.execute("CREATE TABLE IF NOT EXISTS experiments (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, "
                 "status TEXT NOT NULL, payload TEXT NOT NULL)")


def _load(conn, identity):
    row = conn.execute("SELECT payload FROM experiments WHERE id=?", (identity,)).fetchone()
    if not row:
        raise ValueError("没有该实验")
    return json.loads(row[0])


def _save(conn, payload):
    conn.execute("INSERT INTO experiments VALUES (?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                 "status=excluded.status,payload=excluded.payload",
                 (payload["id"], payload["created_at"], payload["status"], store.encoded(payload)))


def register(body, db=store.DB_PATH, at=None):
    """Hypothesis, control, split and attempt budget are fixed before the first run."""
    if not isinstance(body, dict) or any(k not in body for k in REQUIRED):
        raise ValueError(f"实验登记需要：{', '.join(REQUIRED)}")
    split = body["sample_split"]
    if not isinstance(split, dict) or any(k not in split for k in SPLIT_KEYS):
        raise ValueError(f"样本切分需要：{', '.join(SPLIT_KEYS)}")
    if not split["discover_end"] < split["freeze_date"] <= split["oos_start"]:
        raise ValueError("发现段必须早于冻结日，冻结日不得晚于样本外起点")
    if any(word in str(body["metric"]).lower() for word in FORBIDDEN_METRICS):
        raise ValueError("实验指标只允许发现能力口径，不接受收益或胜率")
    if not isinstance(body["max_attempts"], int) or not 1 <= body["max_attempts"] <= 20:
        raise ValueError("尝试次数需为 1 至 20 的整数")
    acceptance = body.get("oos_acceptance")
    if (not isinstance(acceptance, dict) or any(k not in acceptance for k in ECONOMIC_REQUIRED)
            or any(k not in ECONOMIC_CRITERIA for k in acceptance)):
        raise ValueError("须预先冻结样本外经济验收标准：至少 expectancy_net_min 与 n_effective_min；"
                         "可选 max_drawdown_max / tail_loss_max / coverage_min")
    for key, value in acceptance.items():
        _finite(value, key, minimum=1 if key == "n_effective_min" else None, integer=key == "n_effective_min")
    from . import rules
    target = str(body.get("target_part", ""))
    if target in rules.FREEZE["parts"] and not body.get("shadow_only"):
        raise ValueError(f"{target} 已冻结用于前向验证；只能登记影子版本（shadow_only=true），不能改动生效规则")
    at = at or store.now()
    identity = store.digest([body["hypothesis"], body["control"], split])[:16]
    payload = {"id": identity, "version": VERSION, "created_at": at, "status": "REGISTERED",
               "hypothesis": body["hypothesis"], "control": body["control"], "sample_split": split,
               "metric": body["metric"], "max_attempts": body["max_attempts"], "attempts": [],
               "oos_acceptance": acceptance, "target_part": target, "shadow_only": bool(body.get("shadow_only")),
               "economic_check": None,
               "conclusion": None, "decision": None, "rule_versions": body.get("rule_versions", {}),
               "orders_enabled": False}
    with store.connect(db) as conn:
        _table(conn)
        if conn.execute("SELECT 1 FROM experiments WHERE id=?", (identity,)).fetchone():
            return {**_load(conn, identity), "created": False}
        _save(conn, payload)
    return {**payload, "created": True}


def attempt(identity, result, db=store.DB_PATH, at=None):
    """Every run counts against the budget, including the ones that looked bad."""
    with store.connect(db) as conn:
        payload = _load(conn, identity)
        if payload["status"] in {"CONCLUDED", "DECIDED"}:
            raise ValueError("实验已结论，不能再增加尝试")
        if len(payload["attempts"]) >= payload["max_attempts"]:
            raise ValueError("尝试次数已用尽，须重新登记新假设")
        payload["attempts"].append({"n": len(payload["attempts"]) + 1, "at": at or store.now(),
                                    "result": result, "result_sha256": store.digest(result)})
        payload["status"] = "RUNNING"
        _save(conn, payload)
    return payload


def conclude(identity, verdict, evidence, db=store.DB_PATH, at=None):
    if verdict not in {"SUPPORTED", "NOT_SUPPORTED", "INCONCLUSIVE"}:
        raise ValueError("结论只能是 SUPPORTED / NOT_SUPPORTED / INCONCLUSIVE")
    with store.connect(db) as conn:
        payload = _load(conn, identity)
        payload["conclusion"] = {"verdict": verdict, "evidence": evidence, "at": at or store.now(),
                                 "attempts_used": len(payload["attempts"])}
        payload["status"] = "CONCLUDED"
        _save(conn, payload)
    return payload


def economic_check(identity, results, db=store.DB_PATH, at=None):
    """Strategy gate: frozen out-of-sample criteria against results recorded after the freeze.

    Software tests passing says the code has no bug; only this says the rule has economic value.
    The criteria never move to meet the results.
    """
    need = ("n_effective", "expectancy_net", "source", "oos_start")
    if not isinstance(results, dict) or any(k not in results for k in need):
        raise ValueError(f"样本外结果需要：{', '.join(need)}")
    # Evidence is validated before it is compared: a NaN sample count would otherwise slip past every bound.
    _finite(results["n_effective"], "n_effective", minimum=1, integer=True)
    for metric in ECONOMIC_METRICS:
        if metric in results:
            _finite(results[metric], metric)
    if not isinstance(results["source"], str) or not results["source"].strip():
        raise ValueError("样本外结果必须注明来源账本或协议")
    start = _date(results["oos_start"], "oos_start")
    end = _date(results["oos_end"], "oos_end") if results.get("oos_end") else ""
    if end and end < start:
        raise ValueError("样本外区间终点早于起点")
    from . import rules
    with store.connect(db) as conn:
        payload = _load(conn, identity)
        criteria = payload["oos_acceptance"]
        if start < payload["sample_split"]["oos_start"]:
            raise ValueError("样本外结果起点早于登记的样本外起点，不是样本外")
        failed = []
        if results["n_effective"] < criteria["n_effective_min"]:
            status = "INSUFFICIENT_SAMPLE"
            failed.append(f"n_effective={results['n_effective']} < {criteria['n_effective_min']}")
        else:
            rules_ = (("expectancy_net", "expectancy_net_min", lambda v, c: v >= c),
                      ("max_drawdown", "max_drawdown_max", lambda v, c: v <= c),
                      ("tail_loss", "tail_loss_max", lambda v, c: v <= c),
                      ("coverage", "coverage_min", lambda v, c: v >= c))
            for metric, key, ok in rules_:
                if key not in criteria:
                    continue
                if metric not in results:
                    failed.append(f"{metric} 缺失")
                elif not ok(results[metric], criteria[key]):
                    failed.append(f"{metric}={results[metric]} 不满足 {key}={criteria[key]}")
            status = "FAIL" if failed else "PASS"
        check = {"n": len(payload.get("economic_checks") or []) + 1, "status": status, "at": at or store.now(),
                 "criteria": criteria, "results": results, "failed": failed,
                 "oos_window": {"start": start, "end": end}, "source": results["source"].strip(),
                 "evidence_sha256": store.digest(results), "rule_fingerprint": rules.build()["fingerprint"],
                 "basis": "参数在发现段确定后不再调整；样本外经济指标只作验收，不作优化目标"}
        # Every submission stays on record; a later pass never erases an earlier failure.
        payload.setdefault("economic_checks", []).append(check)
        payload["economic_check"] = check
        _save(conn, payload)
    return payload


def decide(identity, decision, reason, db=store.DB_PATH, at=None, effective_from=""):
    """Publish or veto; a publish needs both gates: software (scenarios, replay) and strategy (OOS economics)."""
    if decision not in {"PUBLISHED", "VETOED"}:
        raise ValueError("发布决定只能是 PUBLISHED 或 VETOED")
    with store.connect(db) as conn:
        payload = _load(conn, identity)
        if payload["status"] != "CONCLUDED":
            raise ValueError("未结论的实验不能发布或否决")
        software = (payload.get("publish_check") or {}).get("status")
        strategy = (payload.get("economic_check") or {}).get("status")
        if decision == "PUBLISHED":
            if software != "PASS":
                raise ValueError("软件门禁未通过：固定风险场景与回放对照必须 PASS")
            if strategy != "PASS":
                raise ValueError("策略门禁未通过：样本外经济验收必须 PASS；软件测试通过不等于策略有效")
        payload["decision"] = {"decision": decision, "reason": reason, "at": at or store.now(),
                               "effective_from": effective_from,
                               "gates": {"software": software, "strategy": strategy}}
        payload["status"] = "DECIDED"
        _save(conn, payload)
    return payload


def _fills_digest(db):
    from . import paper
    with paper.database(db) as conn:
        rows = [dict(r) for r in conn.execute("SELECT * FROM paper_fills ORDER BY id")]
    return store.digest(rows), len(rows)


def rollback(identity, reason, db=store.DB_PATH, at=None, effective_from=""):
    """Rollback changes what happens next; the fills that already happened stay bit-identical."""
    before, n_before = _fills_digest(db)
    with store.connect(db) as conn:
        payload = _load(conn, identity)
        if not payload.get("decision") or payload["decision"]["decision"] != "PUBLISHED":
            raise ValueError("只有已发布的实验才能回退")
        payload["decision"] = {**payload["decision"], "decision": "ROLLED_BACK", "rollback_reason": reason,
                               "rolled_back_at": at or store.now(), "rollback_effective_from": effective_from}
        payload["status"] = "DECIDED"
        _save(conn, payload)
    after, n_after = _fills_digest(db)
    return {**payload, "fills_unchanged": before == after and n_before == n_after,
            "note": "回退只影响后续计算；已发生成交未被改写。"}


SCENARIOS = {
    "limit_up_open": {"open_ratio": 1.10, "expect_status": "CANCELLED", "expect_reason": "SECURITY_OR_LIMIT_NOT_FILLABLE"},
    "gap_above_window": {"open_ratio": 1.06, "expect_status": "CANCELLED", "expect_reason": "OPEN_OUTSIDE_FROZEN_WINDOW"},
    "halt_no_bar": {"open_ratio": None, "expect_status": "OPEN", "expect_reason": "NO_VERIFIED_BAR"},
    # A small order fills whole; a larger one fills the lot cap and cancels the IOC remainder.
    "normal_partial_fill": {"open_ratio": 1.02, "expect_status": ("FILLED", "CANCELLED"), "expect_reason": "PAPER_IOC_REMAINDER_CANCELLED"},
    "corporate_action_blocks": {"open_ratio": 1.02, "corporate_action": True, "expect_status": "CANCELLED",
                                "expect_reason": "SECURITY_OR_LIMIT_NOT_FILLABLE"},
}


def _seed_order(copy, at):
    """On the throwaway copy, open a fresh paper order so every scenario has something to replay."""
    from . import paper
    with store.connect(copy) as conn:
        snap = store.latest(conn)
    if not snap:
        return "没有快照可供回放"
    # Approval time must not precede the snapshot's own recording time.
    at = max(at, snap.get("recorded_at", at))
    try:
        with paper.database(copy) as conn:
            if not conn.execute("SELECT 1 FROM paper_account").fetchone():
                return "模拟账户未配置"
            held = {r[0] for r in conn.execute("SELECT symbol FROM paper_lots WHERE remaining>0")}
            pending = {json.loads(r[0])["symbol"] for r in conn.execute(
                "SELECT p.payload FROM paper_orders o JOIN paper_plans p ON p.id=o.plan_id WHERE o.status IN ('OPEN','PARTIAL','UNKNOWN')")}
        symbol = next((r["symbol"] for r in snap["records"] if r.get("paper_plan_eligible")
                       and r["symbol"] not in held | pending), None)
        if not symbol:
            return "没有可复核的候选可供回放"
        result = paper.submit(snap["snapshot_id"], symbol, copy, at=at)
        return "" if result.get("created") else result.get("reason", "计划未创建")
    except (ValueError, paper.Conflict) as exc:
        return str(exc)


def _rows_digest(payload):
    """Row-level identity of a validation result; equal counts with different rows are not the same result."""
    rows = sorted(([r["symbol"], r["opportunity_id"], r["horizon"], r["status"], r.get("return_pct")]
                   for r in payload.get("records", [])), key=lambda r: (r[0], r[1], r[2]))
    return store.digest(rows)


def publish_check(db=store.DB_PATH, at=None, root=None):
    """Fixed risk scenarios replayed on a throwaway copy of the ledger, plus a replay of the last validation.

    A scenario that could not run is not a pass: the check is INCOMPLETE and nothing gets published on it.
    """
    from . import paper, validation
    outcomes = []
    seed_at = (at or store.now())[:10] + "T16:00:00+08:00"
    with tempfile.TemporaryDirectory() as temp:
        for name, spec in SCENARIOS.items():
            copy = Path(temp) / f"{name}.sqlite3"
            shutil.copyfile(db, copy)
            with paper.database(copy) as conn:
                order = conn.execute("SELECT o.*, p.payload AS plan FROM paper_orders o JOIN paper_plans p ON p.id=o.plan_id "
                                     "WHERE o.side='BUY' AND o.status='OPEN' ORDER BY o.created_at LIMIT 1").fetchone()
            if not order:
                reason = _seed_order(copy, seed_at)
                if reason:
                    outcomes.append({"scenario": name, "status": "SKIPPED", "reason": reason})
                    continue
            with paper.database(copy) as conn:
                order = conn.execute("SELECT o.*, p.payload AS plan FROM paper_orders o JOIN paper_plans p ON p.id=o.plan_id "
                                     "WHERE o.side='BUY' AND o.status='OPEN' ORDER BY o.created_at LIMIT 1").fetchone()
                if not order:
                    outcomes.append({"scenario": name, "status": "SKIPPED", "reason": "没有未成交买单可供回放"})
                    continue
                plan = json.loads(order["plan"])
                # The replayed session is the day after the order, as the next-open paper model requires.
                day = (datetime.fromisoformat(order["created_at"][:10]) + timedelta(days=1)).date().isoformat()
                bars = {}
                if spec["open_ratio"] is not None:
                    pre = plan["limit_cents"] / 100
                    bars[plan["symbol"]] = {"date": day, "open": pre * spec["open_ratio"] if name != "normal_partial_fill" else plan["lower_cents"] / 100 * 1.001,
                                            "preclose": pre, "raw_verified": True, "ordinary_mainboard": True,
                                            "corporate_action": bool(spec.get("corporate_action")),
                                            "previous_market_day": plan["entry_day"]}
                paper.process_open(conn, day, bars)
                row = conn.execute("SELECT status FROM paper_orders WHERE id=?", (order["id"],)).fetchone()
                events = [json.loads(r[0]) for r in conn.execute("SELECT payload FROM paper_events ORDER BY rowid DESC LIMIT 5")]
                reasons = {e.get("reason") for e in events}
                expected = spec["expect_status"] if isinstance(spec["expect_status"], tuple) else (spec["expect_status"],)
                ok = row["status"] in expected and (spec["expect_reason"] in reasons or row["status"] in ("OPEN", "FILLED"))
                outcomes.append({"scenario": name, "status": "PASS" if ok else "FAIL",
                                 "order_status": row["status"], "expected": list(expected),
                                 "reasons_seen": sorted(r for r in reasons if r)})
    replay = {"status": "SKIPPED", "reason": "无冻结验证结果可对照"}
    with store.connect(db) as conn:
        latest = store.latest(conn)
    if latest:
        stored = Path(root or store.ROOT) / "results/mainwave_engine" / latest["as_of"].replace("-", "") / "ranking_validation.json"
        if stored.exists():
            recorded = json.loads(stored.read_text(encoding="utf-8-sig"))
            fresh = validation.evaluate(latest["as_of"], db, Path(root or store.ROOT) / "data")
            digests = {"recorded": _rows_digest(recorded), "fresh": _rows_digest(fresh)}
            replay = {"status": "PASS" if digests["recorded"] == digests["fresh"] else "FAIL",
                      "recorded": recorded["summary"], "fresh": fresh["summary"], "digests": digests,
                      "basis": "逐行对照（股票、机会、窗口、状态、收益）；行数相同不算一致"}
    failed = [o for o in outcomes if o["status"] == "FAIL"] + ([replay] if replay["status"] == "FAIL" else [])
    skipped = [o for o in outcomes if o["status"] == "SKIPPED"] + ([replay] if replay["status"] == "SKIPPED" else [])
    return {"version": VERSION, "at": at or store.now(), "scenarios": outcomes, "replay": replay,
            "status": "FAIL" if failed else "INCOMPLETE" if skipped else "PASS",
            "blocking": [f"{o.get('scenario', 'replay')}: {o.get('reason') or o['status']}" for o in failed + skipped],
            "note": "场景在账本副本上回放，生产账本不动；任一必要场景未执行或失败，都不得发布。"}


def attach_check(identity, check, db=store.DB_PATH):
    with store.connect(db) as conn:
        payload = _load(conn, identity)
        payload["publish_check"] = {"status": check["status"], "at": check["at"],
                                    "scenarios": [(o["scenario"], o["status"]) for o in check["scenarios"]],
                                    "replay": check["replay"]["status"]}
        _save(conn, payload)
    return payload


def cadence(today, done=(), schedule=REVIEW_CADENCE):
    """Fixed review dates; a check is a check, not a parameter change."""
    done = set(done)
    overdue = [d for d in schedule if d < today and d not in done]
    upcoming = [d for d in schedule if d >= today and d not in done]
    return {"version": VERSION, "schedule": list(schedule), "done": sorted(done), "overdue": overdue,
            "next": upcoming[0] if upcoming else None,
            "rule": "保持原研究约束；每次检查不等于每次调参。"}


def listing(db=store.DB_PATH):
    with store.connect(db) as conn:
        _table(conn)
        rows = [json.loads(r[0]) for r in conn.execute("SELECT payload FROM experiments ORDER BY created_at")]
    return {"version": VERSION, "experiments": rows}

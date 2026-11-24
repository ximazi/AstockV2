"""Local UI endpoints; no endpoint accepts an arbitrary fill or broker order."""
import json
from datetime import date
from pathlib import Path
from flask import Blueprint, Response, current_app, jsonify, render_template, request
from . import (store, paper, daily, continuity, diagnostics, explain, modes, rules,
               runner, assessment, intraday_link, performance, experiments, migrations,
               corporate_actions, brief, books)

bp = Blueprint("trading", __name__)


def db():
    return current_app.config.get("TRADING_DB", store.DB_PATH)


@bp.errorhandler(ValueError)
def bad_request(exc):
    return jsonify({"error": str(exc)}), 409 if isinstance(exc, paper.Conflict) else 400


@bp.before_request
def guard():
    if request.method == "POST":
        if request.headers.get("Origin", request.host_url.rstrip("/")) != request.host_url.rstrip("/"):
            return jsonify({"error": "不接受跨站写入"}), 403
        if not request.is_json:
            return jsonify({"error": "需要JSON"}), 400


@bp.after_request
def no_cache(response):
    response.headers["Cache-Control"] = "no-store"
    return response


@bp.get("/trading")
def page():
    return render_template("mainwave_trading.html")


@bp.get("/api/trading/limit-theme")
def limit_theme_data():
    day = request.args.get("as_of") or _latest_day()
    if day:
        day = date.fromisoformat(day).isoformat()
        from ..limit_theme.review import STATUSES
        folder = store.ROOT / "results/mainwave_engine" / day.replace("-", "") / "limit_theme"
        pointer = daily.read(folder / "latest.json") if (folder / "latest.json").exists() else {}
        if pointer:
            report = daily.read(Path(pointer["path"]) / "report.json")
            return jsonify({**report, "status_labels": STATUSES})
    return jsonify({"status": "NO_REPORT", "as_of": day, "rows": [], "notice": "尚无该日涨停股概念归因报告"})


@bp.get("/limit-themes")
def limit_theme_page():
    day = request.args.get("as_of") or _latest_day()
    if day:
        day = date.fromisoformat(day).isoformat()
        folder = store.ROOT / "results/mainwave_engine" / day.replace("-", "") / "limit_theme"
        pointer = daily.read(folder / "latest.json") if (folder / "latest.json").exists() else {}
        if pointer and (Path(pointer["path"]) / "report.html").exists():
            return Response((Path(pointer["path"]) / "report.html").read_text(encoding="utf-8"), mimetype="text/html")
    return Response("尚无该日涨停股概念归因报告，请先运行每日任务。", mimetype="text/plain; charset=utf-8", status=404)


@bp.get("/api/trading")
def dashboard():
    with store.connect(db()) as conn:
        selected = request.args.get("snapshot_id")
        snap = store.snapshot(conn, selected) if selected else store.latest(conn)
        history = [dict(r) for r in conn.execute("SELECT id,as_of,created_at FROM snapshots ORDER BY as_of DESC,rowid DESC LIMIT 60")]
        registry = conn.execute("SELECT payload FROM registry_versions WHERE id=?", (snap["knowledge_registry_id"],)).fetchone() if snap else conn.execute("SELECT payload FROM registry_versions ORDER BY rowid DESC LIMIT 1").fetchone()
    result = {"snapshot": None, "snapshots": history, "registry": json.loads(registry[0]) if registry else None,
              "paper": paper.view(db()), "validation": None, "live_orders_enabled": False}
    if snap:
        result["snapshot"] = {**snap, "records": [{k: v for k, v in r.items() if k not in {"evidence", "source_record"}} for r in snap["records"]]}
        path = store.ROOT / "results/mainwave_engine" / snap["as_of"].replace("-", "") / "ranking_validation.json"
        if path.exists():
            result["validation"] = daily.read(path)
    return jsonify(result)


def artefact(day, name):
    """Closure artefacts are read from the frozen day folder, never recomputed on request."""
    path = store.ROOT / "results/mainwave_engine" / day.replace("-", "") / name
    return daily.read(path) if path.exists() else None


@bp.get("/api/trading/preflight")
def startup():
    day = request.args.get("as_of")
    if not day:
        with store.connect(db()) as conn:
            snap = store.latest(conn)
        day = (snap or {}).get("as_of", "")
    return jsonify(artefact(day, "closure_preflight.json") or {"status": "NOT_RUN", "as_of": day})


@bp.get("/api/trading/diagnostics")
def route_cards():
    with store.connect(db()) as conn:
        identity = request.args.get("snapshot_id")
        snap = store.snapshot(conn, identity) if identity else store.latest(conn)
    if not snap:
        return jsonify({"status": "NO_SNAPSHOT", "routes": [], "cards": []})
    stored = artefact(snap["as_of"], "route_diagnostics.json")
    report = stored if stored and stored.get("snapshot_id") == snap["snapshot_id"] else diagnostics.report(snap)
    route = request.args.get("route")
    verdict = request.args.get("verdict")
    cards = [c for c in report["cards"] if c["cards"]]
    if route:
        cards = [c for c in cards if any(x["route"] == route for x in c["cards"])]
    if verdict:
        cards = [c for c in cards if any(x["verdict"] == verdict for x in c["cards"])]
    return jsonify({**report, "cards": cards[:200], "shown": len(cards[:200]), "matched": len(cards)})


@bp.get("/api/trading/validation/audit")
def audit():
    with store.connect(db()) as conn:
        snap = store.latest(conn)
    if not snap:
        return jsonify({"status": "NO_SNAPSHOT"})
    stored = artefact(snap["as_of"], "validation_audit.json")
    if not stored:
        return jsonify({"status": "NOT_RUN", "as_of": snap["as_of"]})
    return jsonify({"status": "READY", "as_of": snap["as_of"], **stored})


@bp.get("/api/trading/intraday")
def intraday():
    day = request.args.get("as_of")
    if not day:
        with store.connect(db()) as conn:
            snap = store.latest(conn)
        day = (snap or {}).get("as_of", "")
    if not day:
        return jsonify({"status": "NO_SNAPSHOT"})
    result = intraday_link.link(day, db())
    limit = min(int(request.args.get("limit", 100)), 500)
    return jsonify({**result, "records": result.get("records", [])[:limit]})


@bp.get("/api/trading/assessment/<identity>")
def ai_assessment(identity):
    """Evidence pack plus stored verdicts; reading never invokes a model."""
    with store.connect(db()) as conn:
        snap = store.latest(conn)
        row = next((r for r in (snap or {}).get("records", []) if r["opportunity_id"] == identity), None)
    if row is None:
        raise ValueError("当前快照没有该机会")
    pack = assessment.bundle(row, snap, diagnostics.card(row), modes.evaluations(row))
    return jsonify({"bundle": pack, "history": assessment.history(identity, db()),
                    "gate": assessment.gate(None, required=False),
                    "injection_policy": assessment.INJECTION_NOTICE})


@bp.post("/api/trading/assessment/<identity>")
def ai_assess(identity):
    """Explicitly invoke local Codex; failure degrades to UNKNOWN without changing trading rules."""
    with store.connect(db()) as conn:
        snap = store.latest(conn)
        row = next((r for r in (snap or {}).get("records", []) if r["opportunity_id"] == identity), None)
    if row is None:
        raise ValueError("当前快照没有该机会")
    pack = assessment.bundle(row, snap, diagnostics.card(row), modes.evaluations(row))
    return jsonify(assessment.configured_assess(pack, db=db()))


@bp.get("/api/trading/assessment/shadow")
def ai_shadow():
    with store.connect(db()) as conn:
        snap = store.latest(conn)
        assessment._table(conn)
        rows = [json.loads(r[0]) for r in conn.execute(
            "SELECT payload FROM ai_assessments WHERE as_of=?", ((snap or {}).get("as_of", ""),))]
    if not snap:
        return jsonify({"status": "NO_SNAPSHOT"})
    return jsonify(assessment.shadow(snap["records"], rows))


@bp.get("/api/trading/run")
def run_state():
    day = request.args.get("as_of")
    if not day:
        with store.connect(db()) as conn:
            snap = store.latest(conn)
        day = (snap or {}).get("as_of", "")
    return jsonify(runner.status(day, db()) if day else {"status": "NO_SNAPSHOT"})


@bp.get("/api/trading/plans/<plan_id>/timeline")
def plan_timeline(plan_id):
    """One plan, in order: approval, orders, fills, adds, valuations and exits."""
    with paper.database(db()) as conn:
        row = conn.execute("SELECT payload FROM paper_plans WHERE id=?", (plan_id,)).fetchone()
        if not row:
            raise ValueError("没有该模拟计划")
        plan = json.loads(row[0])
        orders = [dict(r) for r in conn.execute(
            "SELECT id,side,quantity,filled,status,created_at,payload FROM paper_orders WHERE plan_id=? ORDER BY created_at,id",
            (plan_id,))]
        fills = [dict(r) for r in conn.execute(
            "SELECT f.* FROM paper_fills f JOIN paper_orders o ON o.id=f.order_id WHERE o.plan_id=? ORDER BY f.day,f.id",
            (plan_id,))]
        lots = [dict(r) for r in conn.execute(
            "SELECT entry_day,quantity,remaining,price_cents,entry_fee_cents FROM paper_lots WHERE plan_id=? ORDER BY entry_day,id",
            (plan_id,))]
        valuations = [json.loads(r[0]) for r in conn.execute(
            "SELECT payload FROM paper_valuations WHERE plan_id=? ORDER BY as_of", (plan_id,))]
    events = []
    events.append({"day": plan["entry_day"], "kind": "PLAN_APPROVED", "detail": "批准50%计划预算首仓",
                   "quantity": plan["quantity"], "price_cents": plan["limit_cents"], "filled": False})
    for order in orders:
        body = json.loads(order["payload"])
        events.append({"day": order["created_at"][:10], "kind": f"ORDER_{order['side']}",
                       "detail": body.get("reason", ""), "status": order["status"],
                       "quantity": order["quantity"], "filled_quantity": order["filled"],
                       "tranche": body.get("tranche"), "filled": order["filled"] > 0})
    for fill in fills:
        events.append({"day": fill["day"], "kind": "FILL", "detail": "模拟成交",
                       "quantity": fill["quantity"], "price_cents": fill["price_cents"], "filled": True})
    for valuation in valuations:
        if valuation.get("status") in {"EXIT_REQUIRED", "REDUCE_REQUIRED"}:
            events.append({"day": valuation["day"], "kind": valuation["status"],
                           "detail": valuation.get("reason", ""), "filled": False})
    events.sort(key=lambda e: (e["day"], 0 if e["kind"].startswith("ORDER") else 1))
    return jsonify({"plan": plan, "orders": orders, "lots": lots, "valuations": valuations,
                    "events": events, "dispatch": rules.dispatch(plan, db()),
                    "notice": "有提示未成交不画成成交；未成交委托以状态标注，不计入持仓。"})


def _latest_day():
    with store.connect(db()) as conn:
        snap = store.latest(conn)
    return (snap or {}).get("as_of", "")


def _review_pointer(day):
    folder = store.ROOT / "results/mainwave_engine" / day.replace("-", "") / "eod_review"
    return daily.read(folder / "latest.json") if (folder / "latest.json").exists() else None


@bp.get("/api/trading/eod_review")
def eod_review_view():
    """The after-close review is read from its frozen version folder, never recomputed on request."""
    day = request.args.get("as_of") or _latest_day()
    latest = _review_pointer(day) if day else None
    attempt_path = store.ROOT / "results/mainwave_engine" / day.replace("-", "") / "eod_review/status.json" if day else None
    attempt = daily.read(attempt_path) if attempt_path and attempt_path.exists() else {}
    if not latest:
        return jsonify({"status": "NO_REVIEW", "as_of": day, "latest_attempt": attempt,
                        "note": f"该日没有可用盘后复盘产物；{attempt.get('status', '')} {attempt.get('error', '')}"})
    report = daily.read(Path(latest["path"]) / "report.json")
    stale = attempt.get("status") not in ("SUCCESS", "PARTIAL") or attempt.get("version") != latest.get("version")
    return jsonify({**report, "status": "STALE_REVIEW" if stale else "PARTIAL" if attempt.get("status") == "PARTIAL" else "READY",
                    "latest_attempt": attempt,
                    "note": f"本次复盘 {attempt.get('status', 'UNKNOWN')}，正在查看上次保存版本；{attempt.get('error', '')}" if stale else ""})


@bp.get("/api/trading/eod_review/report.md")
def eod_review_markdown():
    day = request.args.get("as_of") or _latest_day()
    latest = _review_pointer(day) if day else None
    if not latest:
        return Response("该日没有盘后复盘产物", mimetype="text/plain; charset=utf-8", status=404)
    state = eod_review_view().get_json()
    warning = f"> {state['note']}\n\n" if state.get("note") else ""
    return Response(warning + (Path(latest["path"]) / "report.md").read_text(encoding="utf-8"), mimetype="text/markdown; charset=utf-8")


@bp.get("/api/trading/brief")
def daily_brief():
    day = request.args.get("as_of") or _latest_day()
    if not day:
        return jsonify({"status": "NO_SNAPSHOT"})
    stored = artefact(day, "daily_brief.json")
    return jsonify(stored or brief.build(day))


@bp.get("/api/trading/books")
def books_view():
    day = request.args.get("as_of") or _latest_day()
    stored = artefact(day, "books_daily.json") if day else None
    return jsonify(stored or {"status": "NO_RECORD", "books": books.BOOKS, "rule": books.AUTO_RULE})


@bp.post("/api/trading/experiments/<identity>/economic")
def experiment_economic(identity):
    return jsonify(experiments.economic_check(identity, request.get_json(silent=True) or {}, db()))


@bp.get("/api/trading/performance")
def paper_performance():
    day = request.args.get("as_of") or _latest_day()
    if not day:
        return jsonify({"status": "NO_SNAPSHOT"})
    return jsonify(performance.report(day, db()))


@bp.get("/api/trading/experiments")
def experiment_list():
    day = request.args.get("as_of") or _latest_day()
    reviewed = {p.parent.name[:4] + "-" + p.parent.name[4:6] + "-" + p.parent.name[6:]
                for p in (store.ROOT / "results/mainwave_engine").glob("*_v1_*/reviews/*/interpretation.md")}
    return jsonify({**experiments.listing(db()), "cadence": experiments.cadence(day or "", done=sorted(reviewed))})


@bp.post("/api/trading/experiments")
def experiment_register():
    return jsonify(experiments.register(request.get_json(silent=True), db()))


@bp.post("/api/trading/experiments/<identity>/check")
def experiment_check(identity):
    check = experiments.publish_check(db())
    return jsonify(experiments.attach_check(identity, check, db()) | {"check": check})


@bp.post("/api/trading/experiments/<identity>/decide")
def experiment_decide(identity):
    body = request.get_json(silent=True) or {}
    if body.get("decision") == "ROLLED_BACK":
        return jsonify(experiments.rollback(identity, str(body.get("reason", "")), db(),
                                            effective_from=str(body.get("effective_from", ""))))
    return jsonify(experiments.decide(identity, body.get("decision"), str(body.get("reason", "")), db(),
                                      effective_from=str(body.get("effective_from", ""))))


@bp.get("/api/trading/ops")
def ops():
    return jsonify({**migrations.status(db()), "corporate_actions_pending": corporate_actions.pending(db())})


@bp.post("/api/trading/ops/backup")
def ops_backup():
    result = migrations.backup(db())
    return jsonify({**result, "verification": migrations.verify_restore(result["backup"])})


@bp.post("/api/trading/corporate_actions/<identity>/confirm")
def corporate_action_confirm(identity):
    body = request.get_json(silent=True) or {}
    return jsonify(corporate_actions.confirm(identity, body.get("kind"), db(), factor=body.get("factor"),
                                             cash_per_share_cents=body.get("cash_per_share_cents")))


@bp.get("/api/trading/opportunities/<identity>")
def detail(identity):
    with store.connect(db()) as conn:
        sid = request.args.get("snapshot_id")
        snap = store.snapshot(conn, sid) if sid else store.latest(conn)
        row = next((r for r in (snap or {}).get("records", []) if r["opportunity_id"] == identity), None)
        if row is None:
            raise ValueError("当前快照没有该机会")
    payload = {"snapshot_id": snap["snapshot_id"], "recorded_at": snap["recorded_at"], "as_of": snap["as_of"],
               "record": row, **store.history(identity, db()),
               "continuity": continuity.timeline(identity, db()),
               "explanation": explain.explain(row, snap),
               "diagnosis": diagnostics.card(row),
               "modes": modes.evaluations(row)}
    if request.args.get("export") == "1":
        return Response(store.encoded(payload), mimetype="application/json", headers={
            "Content-Disposition": f'attachment; filename="opportunity_{identity}.json"'})
    return jsonify(payload)


@bp.post("/api/trading/opportunities/<identity>/review")
def review(identity):
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or set(body) != {"snapshot_id", "verdict", "note"}:
        raise ValueError("需要快照、复盘状态和说明")
    return jsonify(store.assess(body["snapshot_id"], identity, {"verdict": body["verdict"], "note": body["note"]}, db()))


@bp.post("/api/trading/paper/account")
def account():
    return jsonify(paper.initialize(request.get_json(silent=True), db()))


@bp.post("/api/trading/paper/plans")
def approve():
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or set(body) != {"snapshot_id", "symbol"} or not all(isinstance(v, str) for v in body.values()):
        raise ValueError("需要快照和股票代码，不接受客户端指定价格或数量")
    return jsonify(paper.submit(body["snapshot_id"], body["symbol"], db()))


@bp.post("/api/trading/paper/orders/<identity>/cancel")
def cancel(identity):
    return jsonify(paper.cancel(identity, db()))


@bp.get("/api/trading/export")
def export():
    with store.connect(db()) as conn:
        snaps = [json.loads(r[0]) for r in conn.execute("SELECT payload FROM snapshots ORDER BY as_of,rowid")]
        reviews = [json.loads(r[0]) for r in conn.execute("SELECT payload FROM assessments ORDER BY id")]
        registries = [json.loads(r[0]) for r in conn.execute("SELECT payload FROM registry_versions ORDER BY rowid")]
    payload = {"snapshots": snaps, "assessments": reviews, "registries": registries, "paper": paper.view(db(), full=True),
               "live_orders_enabled": False}
    return Response(store.encoded(payload), mimetype="application/json", headers={
        "Content-Disposition": 'attachment; filename="trading_closure.json"'})


# ---------------------------------------------------------------- 双层复盘报告（fupan v2）
def _fupan_folder(day):
    return store.ROOT / "results/mainwave_engine" / day.replace("-", "") / "fupan"


@bp.get("/api/trading/board-relay")
def board_relay_view():
    from ..board_relay import folder
    from ..eod_review import read_json

    day = request.args.get("as_of") or _latest_day()
    if not day:
        return jsonify({"watch": None, "auction": None, "status": None})
    day = date.fromisoformat(day).isoformat()
    root = store.ROOT / "results/mainwave_engine"
    loc = folder(root, day)
    watch = read_json(loc / "latest.json") or read_json(loc / "replay.json")
    status = read_json(loc / "status.json")
    auction = {}
    if watch and watch.get("as_of") == day and watch.get("mode") == "FORWARD":
        resolved = read_json(loc / "auction_target.json").get("as_of")
        target = date.fromisoformat(resolved or watch["target"]).isoformat()
        item = read_json(folder(root, target) / "auction.json")
        if item.get("as_of") == target and item.get("previous_day", day) == day:
            auction = item
    return jsonify({"watch": watch or None, "auction": auction or None, "status": status or None})


def _fupan_state(day):
    folder = _fupan_folder(day)
    pointer = daily.read(folder / "latest.json") if (folder / "latest.json").exists() else None
    attempt = daily.read(folder / "status.json") if (folder / "status.json").exists() else {}
    return pointer, attempt


@bp.get("/api/trading/fupan")
def fupan_view():
    """The dual-layer review is read from its frozen version folder; the page never rebuilds it on request."""
    day = request.args.get("as_of") or _latest_day()
    pointer, attempt = _fupan_state(day) if day else (None, {})
    if not pointer or not (Path(pointer["path"]) / "report.json").exists():
        return jsonify({"status": "NO_REPORT", "as_of": day, "latest_attempt": attempt,
                        "note": f"该日没有双层复盘报告；{attempt.get('status', '')} {attempt.get('error', '')}".strip()})
    report = daily.read(Path(pointer["path"]) / "report.json")
    packet_stats = daily.read(Path(pointer["path"]) / "packet_stats.json") if (Path(pointer["path"]) / "packet_stats.json").exists() else {}
    query = "?as_of=" + day
    analysis_meta = daily.read(Path(pointer["path"]) / "analysis_meta.json") if (Path(pointer["path"]) / "analysis_meta.json").exists() else {}
    return jsonify({"status": "READY", "as_of": day, "pointer": pointer, "latest_attempt": attempt, "analysis_meta": analysis_meta,
                    "analyzer": {k: v for k, v in (attempt.get("analyzer") or {}).items() if k != "pointer"},
                    "meta": report["meta"], "summary": report["summary"], "metrics": report["metrics"], "issues": report["issues"],
                    "audit": report["audit"], "requests": report["requests"], "packet_stats": packet_stats,
                    "stocks": [{k: st[k] for k in ("id", "symbol", "name", "pinned", "role", "mode", "status", "metrics", "missing",
                                                   "next_check", "notes", "plan", "opportunity_label", "opportunity_reason")} for st in report["stocks"]],
                    "links": {"html": "/api/trading/fupan/report.html" + query, "packet": "/api/trading/fupan/ai_packet.json" + query,
                              "prompt": "/api/trading/fupan/ai_prompt.md" + query, "json": "/api/trading/fupan/report.json" + query}})


def _fupan_file(name, mimetype):
    day = request.args.get("as_of") or _latest_day()
    pointer, _ = _fupan_state(day) if day else (None, {})
    path = Path(pointer["path"]) / name if pointer else None
    if not path or not path.exists():
        return Response("该日没有双层复盘报告", mimetype="text/plain; charset=utf-8", status=404)
    return Response(path.read_text(encoding="utf-8"), mimetype=mimetype)


@bp.get("/api/trading/fupan/report.html")
def fupan_html():
    return _fupan_file("report.html", "text/html; charset=utf-8")


@bp.get("/api/trading/fupan/report.json")
def fupan_json():
    return _fupan_file("report.json", "application/json; charset=utf-8")


@bp.get("/api/trading/fupan/ai_packet.json")
def fupan_packet():
    return _fupan_file("ai_packet.json", "application/json; charset=utf-8")


@bp.get("/api/trading/fupan/ai_prompt.md")
def fupan_prompt():
    return _fupan_file("ai_prompt.md", "text/markdown; charset=utf-8")


@bp.post("/api/trading/fupan/analysis")
def fupan_analysis():
    """Merge an analyzer's JSON as a new report version; facts, prices and statuses cannot change through this door."""
    from ..fupan.build import apply_analysis
    body = request.get_json(silent=True) or {}
    day = body.get("as_of") or request.args.get("as_of") or _latest_day()
    analysis = body.get("analysis", body)
    if not isinstance(analysis, dict) or not day:
        return jsonify({"error": "需要 analysis JSON 对象与交易日"}), 400
    try:
        pointer = apply_analysis(day, analysis, store.ROOT / "results/mainwave_engine")
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"status": "MERGED", "pointer": pointer})


@bp.post("/api/trading/fupan/analyze")
def fupan_analyze():
    """Ask the configured model for its reading of the latest report; a cached answer is reused for an unchanged packet."""
    from ..fupan import analyzer
    body = request.get_json(silent=True) or {}
    day = body.get("as_of") or request.args.get("as_of") or _latest_day()
    if not day:
        return jsonify({"error": "没有交易日"}), 400
    result = analyzer.run(day, store.ROOT / "results/mainwave_engine", force=bool(body.get("force")))
    folder = _fupan_folder(day)
    if (folder / "status.json").exists():
        status = daily.read(folder / "status.json")
        status["analyzer"] = {k: v for k, v in result.items() if k != "pointer"}
        if result.get("pointer"):
            status.update(result["pointer"])
        daily.write(folder / "status.json", status)
    return jsonify(result)

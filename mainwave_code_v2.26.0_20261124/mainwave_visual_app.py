# -*- coding: utf-8 -*-
"""Local visual workbench for main-wave candidate and single-stock analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, send_file
from plotly.offline import get_plotlyjs

from mainwave_engine.stock_analyzer import OUT_ROOT, analyze, candidate_list, latest_report_date
from mainwave_engine import historical_cases as HC
from mainwave_engine.selection_board import selection_list
from mainwave_engine import etf_tracker as ET
from mainwave_engine import review_workbench as RW
from mainwave_engine import review_batch as RB
from mainwave_engine import virtual_portfolio as VP
from mainwave_engine import s4_scan as S4
from mainwave_engine import intraday_review as IR, intraday_decisions as ID
from mainwave_engine.trading.web import bp as trading_blueprint
from mainwave_engine.trading.jev_web import bp as jev_blueprint


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 512 * 1024
app.config['VIRTUAL_PORTFOLIO_DB'] = VP.DB_PATH
app.config['INTRADAY_ROOT'] = IR.ROOT
app.register_blueprint(trading_blueprint)
app.register_blueprint(jev_blueprint)
app.jinja_env.filters['intraday_label'] = ID.label
PLOTLY_JS = get_plotlyjs()


@app.get("/")
def index():
    date = latest_report_date()
    return render_template("mainwave_stock_analyzer.html", report_date=date)


@app.get("/intraday")
@app.get("/api/intraday")
def intraday_dashboard():
    try:
        day = request.args.get("date") or IR.Q.now_china().date().isoformat()
        data = IR.dashboard(day, app.config['INTRADAY_ROOT'])
        response = jsonify(data) if request.path.startswith("/api/") else app.make_response(render_template("mainwave_intraday.html", data=data))
        response.headers['Cache-Control'] = 'no-store'
        return response
    except (ValueError, OSError, KeyError) as exc:
        return jsonify({"error": f"盘中结果不可用：{exc}"}), 400


@app.get("/api/potential-pool")
def potential_pool():
    try:
        day = IR.pd.Timestamp(request.args.get("date") or IR.Q.now_china().date()).date().isoformat()
        result = IR.PP.latest(app.config['INTRADAY_ROOT'], day, IR.Q.now_china())
        response = jsonify(result or {"as_of": day, "status": "UNAVAILABLE", "records": []})
        response.headers['Cache-Control'] = 'no-store'
        return response
    except (ValueError, OSError, KeyError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/vendor/plotly.min.js")
def plotly_js():
    return Response(PLOTLY_JS, mimetype="application/javascript")


@app.get("/api/candidates")
def candidates():
    try:
        return jsonify(candidate_list(request.args.get("date", "")))
    except (ValueError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/api/s4")
def s4_opportunities():
    """Today's S4 list (s4_scan output) for the top of 每日候选; missing file is reported, never faked."""
    day = request.args.get("date", "").replace("-", "")
    path = OUT_ROOT / day / "s4_opportunities.json" if day else None
    if not path or not path.exists():
        return jsonify({"missing": True, "as_of": day, "recommended": [], "excluded": []})
    return jsonify(json.loads(path.read_text(encoding="utf-8")))


SCENARIO_FILES = {"s4": "s4_opportunities.json", "s3": "s3_observations.json",
                  **{k: f"{k}_observations.json" for k in ("s1", "s2", "s5")}}
SCENARIO_CASES = {"s4": "results/research/s4_features/s4_cases.html", "s3": "results/research/s3_cases/s3_cases.html",
                  **{k: f"results/research/scenario_cases/{k}_cases.html" for k in ("s1", "s2", "s5")}}


@app.get("/api/scenario/<kind>")
def scenario(kind):
    """A scenario's daily list (s4_scan / s3_scan output) for its own menu; a missing day is reported, never faked."""
    if kind not in SCENARIO_FILES:
        return jsonify({"error": "未知场景"}), 404
    day = request.args.get("date", "").replace("-", "")
    path = OUT_ROOT / day / SCENARIO_FILES[kind] if day else None
    if not path or not path.exists():
        return jsonify({"missing": True, "as_of": day})
    return jsonify(json.loads(path.read_text(encoding="utf-8")))


@app.get("/api/scenario-export")
def scenario_export():
    """The day's S1-S5 opportunities as one CSV (scenario_scan.export over the files each scan wrote)."""
    from mainwave_engine import scenario_scan as SC
    day = request.args.get("date", "")
    if len(day.replace("-", "")) != 8 or not day.replace("-", "").isdigit():
        return jsonify({"error": "需要日期 YYYY-MM-DD"}), 400
    day = f"{day.replace('-', '')[:4]}-{day.replace('-', '')[4:6]}-{day.replace('-', '')[6:]}"
    path, _, _ = SC.export(day, OUT_ROOT / day.replace("-", ""))
    return send_file(Path(path).resolve(), mimetype="text/csv", as_attachment=True, download_name=Path(path).name)


@app.get("/scenario-cases/<kind>")
def scenario_cases(kind):
    """The scenario's case gallery (10 winners / 10 losers, K-line + MACD)."""
    if kind not in SCENARIO_CASES or not Path(SCENARIO_CASES[kind]).exists():
        return jsonify({"error": "案例图谱尚未生成"}), 404
    return send_file(Path(SCENARIO_CASES[kind]).resolve())


@app.get("/api/analyze")
def analyze_stock():
    try:
        result = analyze(
            request.args.get("symbol", ""), request.args.get("date", ""), persist=True
        )
        return jsonify(result)
    except (ValueError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # The UI must return a usable error instead of an HTML traceback.
        return jsonify({"error": f"分析失败：{exc}"}), 500


@app.get("/api/virtual-positions")
def virtual_positions():
    return jsonify(VP.listing(app.config['VIRTUAL_PORTFOLIO_DB']))


@app.post("/api/virtual-positions")
@app.post("/api/virtual-positions/<position_id>/close")
def virtual_position_write(position_id=None):
    if request.headers.get("Origin", request.host_url.rstrip("/")) != request.host_url.rstrip("/"):
        return jsonify({"error": "不接受跨站持仓写入"}), 403
    body = request.get_json(silent=True)
    fields = {"date"} if position_id else {"symbol", "date"}
    allowed = (fields,) if position_id else (fields, fields | {"strategy"})
    if not isinstance(body, dict) or set(body) not in allowed or any(not isinstance(v, str) for v in body.values()):
        return jsonify({"error": "需要股票代码和行情日期；收盘价由服务端读取"}), 400
    try:
        db = app.config['VIRTUAL_PORTFOLIO_DB']
        if position_id:
            return jsonify(VP.close_position(position_id, body['date'], db))
        if body.get('strategy') == 'S4_LAD':  # only names on that day's S4 list
            return jsonify(S4.pick(body['date'], body['symbol'], db=db))
        return jsonify(VP.add(body['symbol'], body['date'], db, strategy=body.get('strategy')))
    except VP.Conflict as exc:
        return jsonify({"error": str(exc)}), 409
    except (ValueError, OSError, KeyError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/api/history-cases")
def history_cases():
    return jsonify(HC.case_list())


@app.get("/api/selections")
def selections():
    try:
        return jsonify(selection_list())
    except (ValueError, FileNotFoundError) as exc:
        return jsonify({"error": f"筛选榜单读取失败：{exc}"}), 500


@app.get("/api/history-cases/<case_id>")
def history_case(case_id):
    try:
        return jsonify(HC.case_detail(case_id))
    except (ValueError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/api/etfs")
def etfs():
    try:
        return jsonify(ET.watch_list(request.args.get("date", "")))
    except (ValueError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/api/etfs/<code>")
def etf_detail(code):
    try:
        return jsonify(ET.detail(code, request.args.get("date", "")))
    except (ValueError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/api/reviews")
def review_list():
    try:
        return jsonify(RW.case_list())
    except (ValueError, OSError) as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/api/review-exports")
def review_batch_list():
    paths = sorted((RW.ROOT / "exports").glob("*/status.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:10]
    return jsonify([RB.status(p.parent.name) for p in paths])


@app.post("/api/review-exports")
def review_batch_start():
    if request.headers.get("Origin", request.host_url.rstrip("/")) != request.host_url.rstrip("/"):
        return jsonify({"error": "不接受跨站导出请求"}), 403
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or set(body) != {"case_ids", "kind"}:
        return jsonify({"error": "需要案例编号列表和导出类型"}), 400
    try:
        return jsonify(RB.start(body["case_ids"], body["kind"])), 202
    except RW.Conflict as exc:
        return jsonify({"error": str(exc)}), 409
    except (ValueError, OSError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/api/review-exports/<job_id>")
def review_batch_status(job_id):
    try:
        return jsonify(RB.status(job_id))
    except (ValueError, OSError) as exc:
        return jsonify({"error": str(exc)}), 404


@app.post("/api/review-exports/<job_id>/cancel")
def review_batch_cancel(job_id):
    if request.headers.get("Origin", request.host_url.rstrip("/")) != request.host_url.rstrip("/"):
        return jsonify({"error": "不接受跨站取消请求"}), 403
    try:
        return jsonify(RB.cancel(job_id))
    except (ValueError, OSError) as exc:
        return jsonify({"error": str(exc)}), 404


@app.get("/api/review-exports/<job_id>/download")
def review_batch_download(job_id):
    try:
        state = RB.status(job_id)
        if not state.get("download_url"):
            return jsonify({"error": "导出包尚未完成"}), 409
        return send_file(RB.job_dir(job_id, RW.ROOT) / "bundle.zip", as_attachment=True,
                         download_name=f"reviews_{state['kind']}_{job_id[:8]}.zip", max_age=0)
    except (ValueError, OSError) as exc:
        return jsonify({"error": str(exc)}), 404


@app.get("/api/reviews/<case_id>")
def review_detail(case_id):
    try:
        from mainwave_engine.review_catalog import selection_for
        from mainwave_engine.entry_exit_study import load
        from mainwave_engine.price_volume_context import case_attachment
        case = RW.detail(case_id)
        return jsonify({"case": case, "review": RW.load_review(case_id), "stages": RW.STAGES,
                        "exit_selection": selection_for(case_id), "entry_exit_study": load(case["summary"]),
                        "price_volume_context": case_attachment(case)})
    except (ValueError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 404


@app.get("/api/reviews/<case_id>/entry-exit-study")
def review_entry_exit_study(case_id):
    try:
        from mainwave_engine.entry_exit_study import load
        study = load(RW.detail(case_id)["summary"])
        if study is None:
            raise FileNotFoundError("该案例尚无完成的买卖点研究")
        return Response(RW.encoded(study), mimetype="application/json", headers={
            "Content-Disposition": f'attachment; filename="{case_id}_entry_exit_study.json"',
            "Cache-Control": "no-store"})
    except (ValueError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 404


@app.post("/api/reviews/<case_id>")
def review_save(case_id):
    if request.headers.get("Origin", request.host_url.rstrip("/")) != request.host_url.rstrip("/"):
        return jsonify({"error": "不接受跨站分析写入"}), 403
    if not request.is_json:
        return jsonify({"error": "需要JSON分析记录"}), 400
    try:
        return jsonify(RW.save_review(case_id, request.get_json(silent=True)))
    except RW.Conflict as exc:
        return jsonify({"error": str(exc)}), 409
    except (ValueError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/api/reviews/<case_id>/export/<kind>")
def review_export(case_id, kind):
    try:
        data = RW.archive(case_id, kind)
        return Response(data, mimetype="application/zip", headers={
            "Content-Disposition": f'attachment; filename="{case_id}_{kind}.zip"',
            "Cache-Control": "no-store"})
    except (ValueError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 404


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8068)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

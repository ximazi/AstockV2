"""Frozen backtest cases and separately versioned human reviews. No trading writes."""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import date, datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import sqlite3
import zipfile

import pandas as pd

from . import __version__, stock_analyzer as SA, stock_lifecycle as SL
from . import price_cycle_setups as PCS, route_evidence as RE

ROOT = Path(__file__).resolve().parents[1] / "results/mainwave_engine/review_workbench"
SOURCE = Path(__file__).resolve().parents[1] / "results/mainwave_engine/route_diagnostic_20260910"
VERSION = "review_workbench_v1"
NOTICE = "系统历史回测记录，不是当时生产命中或实盘成交。生命周期由当前引擎因果重建，人工结论独立保存；一例不能代表胜率。"
TEXT_FIELDS = ("reviewer", "summary", "lifecycle_notes", "entry_notes", "exit_notes", "ai_notes", "improvements")
STAGES = {**SL.LABELS, "UNCERTAIN": "无法判断"}


class Conflict(ValueError):
    pass


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def records(frame):
    frame = frame.copy()
    if "date" in frame:
        frame["date"] = frame.date.astype(str).str[:10]
    return json.loads(frame.to_json(orient="records", force_ascii=False, double_precision=15))


def detail(case_id, root=ROOT):
    if not re.fullmatch(r"[a-zA-Z0-9_]{1,100}", case_id):
        raise ValueError("案例编号无效")
    path = root / "cases" / f"{case_id}.json"
    if not path.is_file() and (root / "catalog.json").exists():
        from .review_catalog import materialize
        materialize(case_id, root)
    if not path.is_file():
        raise FileNotFoundError("复盘案例不存在")
    case = json.loads(path.read_text(encoding="utf-8"))
    if case.get("case_hash") != digest(encoded({k: v for k, v in case.items() if k != "case_hash"})):
        raise ValueError("案例快照校验失败，不能继续分析或导出")
    return case


def case_list(root=ROOT):
    if (root / "catalog.json").exists():
        from .review_catalog import listing
        return listing(root)
    rows = []
    for path in sorted((root / "cases").glob("*.json")):
        case = detail(path.stem, root)
        review = load_review(path.stem, root)
        rows.append({**case["summary"], "revision": review["revision"], "review_status": review["status"]})
    from .review_catalog import review_order
    rows.sort(key=review_order)
    return {"cases": rows, "notice": NOTICE, "stages": STAGES,
            "scope": f"已归档{len(rows)}个研究案例，含事后选取的亏损诊断样本；非全量清单，不据此计算胜率"}


def seed_2026(root=ROOT, source=SOURCE):
    """Pick the first signal, not a winner; repeated invocation never overwrites a case."""
    frame_path = source / "r1_exit_replay.csv.gz"
    all_rows = pd.read_csv(frame_path)
    selected = all_rows[all_rows.year.eq(2026) & all_rows.exit.eq("legacy20")].sort_values(["date", "symbol"])
    if selected.empty:
        raise ValueError("没有2026年回测信号，不生成演示案例")
    row = records(selected.head(1))[0]
    return import_r1_case(row, "2026年legacy20回测信号按日期、代码排序第一条；未按收益筛选。", root, source)


def failure_samples(all_rows):
    """Diagnostic outcome sampling only, never a production filter or win-rate cohort."""
    losses = all_rows[all_rows.year.eq(2026) & all_rows.exit.eq("legacy20") &
                      all_rows.status.eq("CLOSED") & all_rows.net.lt(0)].sort_values(["date", "symbol"])
    groups = (
        ("快速退场", losses.reason.eq("TRAIL_STOP") & losses.held.le(3) & losses.mfe.lt(2),
         "跟踪止损退出、持有不超过3日、持有期MFE低于2%"),
        ("冲高回吐", losses.mfe.ge(8), "持有期MFE不低于8%但最终净亏损"),
        ("止损延后", losses.reason.eq("DEFERRED_STOP"), "回测退出原因为DEFERRED_STOP"),
    )
    selected, used = [], set()
    for label, mask, rule in groups:
        candidates = losses[mask & ~losses.symbol.isin(used)]
        if candidates.empty:
            raise ValueError(f"缺少可核验的{label}样本，不补造案例")
        row = records(candidates.head(1))[0]
        used.add(row["symbol"])
        selection = (f"亏损诊断样本：{label}。母集为2026年legacy20已平仓且净收益<0的回测记录；"
                     f"本类条件：{rule}。每类按信号日期、代码取第一条，跨类股票去重。"
                     "按事后交易结果选例，仅用于复盘；分类阈值不是买卖规则，不代表失败因果或总体胜率。")
        selected.append((row, selection, label))
    return selected


def seed_failures_2026(root=ROOT, source=SOURCE):
    samples = failure_samples(pd.read_csv(source / "r1_exit_replay.csv.gz"))
    return [import_r1_case(row, selection, root, source, label) for row, selection, label in samples]


def import_r1_case(row, selection, root=ROOT, source=SOURCE, sample_label=""):
    from research.route_followup import arrays, enrich, simulate
    from research import route_followup as RF, route_clock as CLOCK

    from .review_catalog import POLICIES
    if row["year"] != 2026 or row["exit"] not in POLICIES:
        raise ValueError("当前导入器仅支持2026年R1已知退出规则")
    frame_path = source / "r1_exit_replay.csv.gz"
    symbol, signal = row["symbol"], row["date"]
    policy = row["exit"]
    case_id = f"R1_{symbol}_{signal.replace('-', '')}_{policy}"
    if (root / "cases" / f"{case_id}.json").exists():
        return detail(case_id, root)["summary"]
    manifest = json.loads((source / "exit_replay_manifest.json").read_text(encoding="utf-8"))
    as_of = manifest["as_of"]
    price_path = (source / "data" if (source / "data").exists() else Path("data")) / f"{symbol}_daily.csv"
    price_hash = digest(price_path.read_bytes())
    original_hash = next(r["price_hash"] for r in manifest["inputs"] if r["symbol"] == symbol)
    if price_hash != original_hash:
        raise ValueError("行情文件与原回测输入哈希不同，先核验差异再导入案例")
    raw_source = pd.read_csv(price_path)
    raw_source = raw_source[pd.to_datetime(raw_source.date).le(as_of)]
    evidence = enrich(RE.build(raw_source, symbol))
    a = arrays(evidence)
    signal_i = list(a["day"]).index(signal)
    if not bool(evidence.iloc[signal_i].accepted_bias2):
        raise ValueError("原始信号不能按当前证据重现")
    trace = []
    replay = simulate(a, signal_i, symbol, policy, trace=trace,
                      calendar=tuple(manifest["frozen_calendar"]) if "frozen_calendar" in manifest else None)
    for key in ("status", "entry_date", "exit_date", "reason"):
        if row.get(key) != replay.get(key):
            raise ValueError(f"交易复现不一致：{key}")
    for key in ("net", "gross", "entry_price"):
        if row.get(key) is not None and abs(row[key] - replay[key]) > 1e-8:
            raise ValueError(f"交易价格/收益复现不一致：{key}")
    chart_raw = raw_source.copy()
    chart_raw["date"] = pd.to_datetime(chart_raw.date)
    chart_raw = chart_raw.sort_values("date").drop_duplicates("date", keep="last")
    chart_raw = chart_raw[pd.to_numeric(chart_raw.amount, errors="coerce") > 0].reset_index(drop=True)
    classified = SA.PE.classify_frame(chart_raw)
    timeline = SL.replay(symbol, classified)
    dates = timeline.date.astype(str).str[:10]
    i = list(dates).index(signal)
    start_i = max(0, i - 90)
    cid = timeline.iloc[i].cycle_id
    if cid:
        first_cycle_i = next(j for j, value in enumerate(timeline.cycle_id) if value == cid)
        start_i = min(start_i, max(0, first_cycle_i - 25))
    window_start = str(dates.iloc[start_i])
    visible = timeline.iloc[start_i:]
    bars = SA._adjusted_chart(classified, len(visible))
    ev = evidence[evidence.date.ge(window_start)].copy()
    lc = visible.copy()
    lc["date"] = lc.date.astype(str).str[:10]
    ev["date"] = ev.date.astype(str).str[:10]
    daily = lc.merge(ev, on="date", how="left", suffixes=("", "_evidence"), validate="one_to_one")
    anchor = float(classified.iloc[-1].close / classified.iloc[-1].return_index)
    marks = [{"kind": "TARGET_IGNITION", "date": signal, "label": "R1价格接受信号（点火代理）",
              "chart_price": float(a["index_close"][signal_i]) * anchor, "raw_price": float(a["close"][signal_i]),
              "signal_at": signal, "reason": "accepted_bias2价格确认，非题材点火"}]
    if trace:
        marks.append({"kind": "BUY", "date": trace[0]["date"], "label": "回测买入", "chart_price": trace[0]["entry_index"] * anchor,
                      "raw_price": float(a["open"][list(a["day"]).index(trace[0]["date"])]), "signal_at": signal, "reason": "原始回测规则：收盘信号后的下一开盘"})
        for day in trace:
            k = list(a["day"]).index(day["date"])
            for fill in day["fills"]:
                marks.append({"kind": "SELL", "date": day["date"], "label": "回测卖出", "chart_price": fill["price_index"] * anchor,
                              "raw_price": fill["price_index"] * float(a["close"][k] / a["index_close"][k]), **fill})
    mapping = SA._mapping(symbol)
    summary = {"id": case_id, "symbol": symbol, "name": mapping.get("name", symbol), "route": "R1",
               "signal_date": signal, "window_start": window_start, "window_end": bars[-1]["date"],
               "trade_status": row["status"], "net_pct": row.get("net"), "exit_policy": policy,
               "sample_label": sample_label}
    case = {"schema_version": VERSION, "summary": summary, "notice": NOTICE,
            "created_at": datetime.now(timezone.utc).isoformat(), "engine_version": __version__,
            "selection": selection, "source_row": row,
            "source": {"file": str(frame_path), "sha256": digest(frame_path.read_bytes()), "price_sha256": price_hash,
                       "manifest": manifest | {"inputs": [r for r in manifest["inputs"] if r["symbol"] == symbol]},
                       "production_hit_proven": False, "replay_matched": True},
            "reconstruction": {"as_of": as_of, "cycle_version": SL.VERSION, "evidence_version": RE.VERSION,
                               "code_sha256": {Path(m.__file__).name: digest(Path(m.__file__).read_bytes()) for m in (SL, PCS, RE, RE.PE, RE.PT, RE.BE, RF, CLOCK)},
                               "warmup_start": str(raw_source.date.min()), "daily_rows": len(daily),
                               "target_cycle_id": cid, "target_cycle_failed_at": next((str(r.date) for _, r in visible.iterrows() if r.cycle_id == cid and r.cycle_stage == "FAILED"), None),
                               "boundary_note": "含信号前至少90个活跃交易日背景，展示至冻结行情截止日；退出后继续观察，不以卖点截断。未观测到的阶段不补造。"},
            "units": {"raw_ohlc": "原始人民币价格", "chart_ohlc": "以窗口末原始收盘为锚的等比复权价格",
                      "index_prices": "return_index尺度，不能当作人民币", "amount": "元；图中成交额为亿元", "turnover": "%",
                      "net_gross_mfe_mae": "百分比；固定总成本0.30个百分点；非账户收益"},
            "rules": {"signal": "accepted_bias2：当日有效价格接受 + 需求证据 + 无供给警告 + (复权价-MA20)/ATR20 <= 2",
                      "entry": "收盘信号后的下一交易所会话开盘；涨停开盘、ST或会话数据无法核验时不假定成交",
                      "initial_stop": "买入价 - 2 * 信号日ATR20，使用return_index统一尺度",
                      "legacy20_trail": "入场后的第二个活跃日开始，收盘更新：max(原保护价, 持有以来最近至多10根K线最低价)；翌日执行",
                      "exit": "T+1可卖后触及保护价，按min(当日开盘,保护价)模拟；跌停锁死顺延；最多持有20个活跃交易日，末日未锁死则按收盘退出",
                      "cost": "总成本固定扣0.30个百分点，不是逐笔真实佣金；非组合、非实盘，未包含完整历史市场和题材许可"},
            "daily_field_groups": {"cycle_*": "连续价格周期及变化原因；原始系统输出，不被人工标注覆盖",
                                   "setup_*": "同一突破链上的回踩/整理事件及冻结价格", "preparation_*": "资金准备代理身份、起点、有效性；不是主力账户证明",
                                   "breakout_*": "突破尝试、接受、推进、失效及可知日期", "*_evidence": "与周期输出同名的独立路线证据列",
                                   "ret5/ret20/ret60": "小数收益率；与net/gross百分比单位不同", "trace.stop_before/stop_next": "当日执行保护价/次日生效保护价，均为return_index尺度"},
            "daily": records(daily), "execution_trace": trace, "replay_result": replay,
            "raw_history": records(raw_source),
            "view": {"meta": {"mode": "HISTORICAL_RECONSTRUCTION", "symbol": symbol}, "identity": mapping,
                     "chart": {"bars": bars, "stage_segments": PCS.segments(visible), "price_basis": "窗口末原始收盘锚定的等比复权价格"},
                     "lifecycle": {"history": records(visible)}, "event": {"history": SA._event_history(classified, len(visible))},
                     "setups": {"history": records(PCS.setup_history(visible))}, "case_review": {"events": marks}}}
    if policy != "legacy20":
        case["rules"].pop("legacy20_trail")
        case["rules"]["trail"] = "持有期最高价达到入场价+2ATR后激活；收盘上移至最近10根K线最低价，翌日生效，保护价不下降"
        case["rules"]["partials"] = "T+1可卖且未跌停锁死时，+3ATR退出原始仓位50%，+6ATR退出25%；保护止损优先于同日目标；剩余按保护或20日期限退出"
        if policy == "structure36_20":
            case["rules"]["initial_stop"] = "max(买入价-2*信号日ATR20, 信号日冻结结构失效位)；入场开盘不高于保护位则不成交"
    case["case_hash"] = digest(encoded(case))
    target = root / "cases" / f"{case_id}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        handle.write(encoded(case))
    return summary


def blank_review(case):
    return {"case_id": case["summary"]["id"], "case_hash": case["case_hash"], "revision": 0,
            "status": "DRAFT", "verdict": "UNCERTAIN", "updated_at": None,
            **{key: "" for key in TEXT_FIELDS}, "annotations": []}


def saved_reviews(root=ROOT):
    if not (root / "reviews.sqlite3").exists():
        return {}
    with closing(sqlite3.connect(root / "reviews.sqlite3")) as conn:
        rows = conn.execute("SELECT r.case_id,r.payload FROM reviews r JOIN (SELECT case_id,MAX(revision) AS revision FROM reviews GROUP BY case_id) latest USING(case_id,revision)").fetchall()
    return {cid: json.loads(payload) for cid, payload in rows}


def load_review(case_id, root=ROOT):
    case = detail(case_id, root)
    db = root / "reviews.sqlite3"
    if db.exists():
        with closing(sqlite3.connect(db)) as conn:
            row = conn.execute("SELECT payload FROM reviews WHERE case_id=? ORDER BY revision DESC LIMIT 1", (case_id,)).fetchone()
        if row:
            return json.loads(row[0])
    return blank_review(case)


def validate_review(body, case):
    allowed = set(blank_review(case)) - {"updated_at"}
    if not isinstance(body, dict) or set(body) - allowed:
        raise ValueError("分析字段格式不正确")
    if body.get("case_id") != case["summary"]["id"] or body.get("case_hash") != case["case_hash"]:
        raise Conflict("案例版本不一致，请重新载入")
    if type(body.get("revision")) is not int or body["revision"] < 0:
        raise ValueError("分析版本无效")
    if not isinstance(body.get("status"), str) or not isinstance(body.get("verdict"), str) or body["status"] not in {"DRAFT", "REVIEWED"} or body["verdict"] not in {"AGREE", "DISAGREE", "PARTIAL", "UNCERTAIN"}:
        raise ValueError("复盘状态或判断无效")
    for key in TEXT_FIELDS:
        if not isinstance(body.get(key), str) or len(body[key]) > (100 if key == "reviewer" else 20000):
            raise ValueError(f"{key}必须是长度合规的文本")
    if body["status"] == "REVIEWED" and not body["summary"].strip():
        raise ValueError("完成复盘前请填写总结")
    annotations = body.get("annotations")
    if not isinstance(annotations, list) or len(annotations) > 100:
        raise ValueError("阶段标注最多100条")
    for item in annotations:
        if not isinstance(item, dict) or set(item) != {"start", "end", "stage", "note"}:
            raise ValueError("阶段标注格式不正确")
        try:
            start, end = date.fromisoformat(item["start"]), date.fromisoformat(item["end"])
        except (ValueError, TypeError):
            raise ValueError("阶段标注日期无效") from None
        bounds = case["summary"]
        if not bounds["window_start"] <= start.isoformat() <= end.isoformat() <= bounds["window_end"]:
            raise ValueError("阶段日期必须在案例观察区间内且起日不晚于止日")
        if not isinstance(item["stage"], str) or item["stage"] not in STAGES or not isinstance(item["note"], str) or len(item["note"]) > 5000:
            raise ValueError("人工阶段或依据无效")
    return dict(body)


def save_review(case_id, body, root=ROOT):
    case = detail(case_id, root)
    value = validate_review(body, case)
    with closing(sqlite3.connect(root / "reviews.sqlite3", timeout=10)) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS reviews (case_id TEXT NOT NULL, revision INTEGER NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(case_id, revision))")
        conn.execute("BEGIN IMMEDIATE")
        revision = conn.execute("SELECT COALESCE(MAX(revision), 0) FROM reviews WHERE case_id=?", (case_id,)).fetchone()[0]
        if revision != value["revision"]:
            raise Conflict("分析已在其他页面更新，请导出当前草稿并重新载入后合并")
        value.update(revision=revision + 1, updated_at=datetime.now(timezone.utc).isoformat())
        conn.execute("INSERT INTO reviews VALUES (?, ?, ?)", (case_id, value["revision"], encoded(value).decode("utf-8")))
        conn.commit()
    return value


def archive(case_id, kind, root=ROOT):
    case = detail(case_id, root)
    if kind not in {"case", "review"}:
        raise ValueError("导出类型无效")
    s = case["summary"]
    title = f"# {s['name']} {s['symbol']} / {s['signal_date']} / {s['route']}\n\n"
    intro = title + NOTICE + f"\n\n案例ID：{case_id}\n\n案例SHA256：{case['case_hash']}\n\n观察区间：{s['window_start']} 至 {s['window_end']}\n\n"
    files = {}
    if kind == "case":
        from .review_chart import chart_files
        from .price_volume_context import case_attachment
        observation = case_attachment(case)
        files.update(chart_files(case, observation=observation))
        files["price_volume_context.json"] = encoded(observation)
        files["price_volume_timeline.csv"] = pd.DataFrame(observation["timeline"]).to_csv(index=False).encode("utf-8-sig")
        files["case.json"] = encoded(case)
        files["lifecycle_daily.csv"] = pd.DataFrame(case["daily"]).to_csv(index=False).encode("utf-8-sig")
        files["raw_history.csv"] = pd.DataFrame(case["raw_history"]).to_csv(index=False).encode("utf-8-sig")
        files["execution_trace.json"] = encoded(case["execution_trace"])
        files["README.md"] = (intro + case["selection"] + "\n\n## 字段与复盘边界\n\n" + "\n".join(f"- {k}: {v}" for k, v in case["units"].items()) + "\n\n## 原始回测规则\n\n" + "\n".join(f"- {k}: {v}" for k, v in case.get("rules", {}).items()) + "\n\ncase.json包含来源行、规则版本、行情哈希、逐日证据与图表；raw_history包含指标预热及停牌原始行。人工分析单独导出，不改变系统原始阶段。\n").encode("utf-8")
        files["README.md"] += "\n## K线图片\n\nkline_full.png：完整观察窗口。kline_trade_focus.png：覆盖全部点火/买卖事件，前后各保留最多25个活跃交易日。两图均为2560×1440 PNG，含阶段背景、均线、成交额和日期/原价标注。图中箭头使用复权位置，不把原始价格直接画到复权K线上。chart_exports.json记录图像窗口与案例哈希。图片从冻结案例生成，不受当前屏幕、缩放或人工标注影响。\n".encode("utf-8")
        files["README.md"] += ("\n## 量价观察附件\n\n" + observation["notice"] + observation["source_notice"]
                               + "\nprice_volume_context.json含新规则、来源哈希和逐日证据；price_volume_timeline.csv便于复盘。K线下方窄条为同版只读观察，灰色代表未明确分类。\n").encode("utf-8")
    else:
        review = load_review(case_id, root)
        revisions = []
        if review["revision"]:
            with closing(sqlite3.connect(root / "reviews.sqlite3")) as conn:
                revisions = [json.loads(r[0]) for r in conn.execute("SELECT payload FROM reviews WHERE case_id=? ORDER BY revision", (case_id,))]
        files["review.json"] = encoded(review)
        files["review_history.json"] = encoded(revisions)
        files["case_reference.json"] = encoded({"summary": s, "case_hash": case["case_hash"], "source_row": case["source_row"], "notice": NOTICE})
        labels = {"reviewer": "复盘人", "summary": "复盘总结", "lifecycle_notes": "生命周期判断", "entry_notes": "入场判断", "exit_notes": "退出判断", "ai_notes": "AI分析记录（未自动核验）", "improvements": "改进假设与待验证项"}
        md = intro + f"保存版本：{review['revision']} / {review['status']} / {review['verdict']}\n\n"
        md += "\n\n".join(f"## {label}\n\n{review[key] or '未填写'}" for key, label in labels.items())
        md += "\n\n## 人工阶段标注\n\n" + "\n\n".join(f"{r['start']} 至 {r['end']}：{STAGES[r['stage']]}\n\n{r['note']}" for r in review["annotations"])
        files["review.md"] = md.encode("utf-8")
    from .review_catalog import selection_for
    selection = selection_for(case_id, root)
    if selection:
        if kind == "case":
            selection = {k: v for k, v in selection.items() if k != "related_reviews"}
        files["exit_selection.json"] = encoded(selection)
        note = f"\n\n## 同信号择优口径\n\n{selection['label']}。{selection['notice']}\n\n展示保留案例：{selection['selected_id']}。原快照及其哈希未改写；本文件记录当前列表的事后择优依据。\n"
        files["README.md" if kind == "case" else "review.md"] += note.encode("utf-8")
        if kind == "review":
            saved = saved_reviews(root)
            related = [saved[r["id"]] for r in selection["alternatives"] if r["id"] != case_id and r["id"] in saved]
            files["other_policy_reviews.json"] = encoded(related)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name, data in files.items():
            bundle.writestr(name, data)
    return stream.getvalue()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--seed-2026", action="store_true")
    mode.add_argument("--seed-failures-2026", action="store_true")
    mode.add_argument("--index-2026", action="store_true")
    args = parser.parse_args()
    if args.index_2026:
        from .review_catalog import build
        catalog = build()
        print(json.dumps({"signals": catalog["signals"], "cases": len(catalog["entries"]), "catalog_hash": catalog["catalog_hash"]}))
    else:
        print(json.dumps(seed_failures_2026() if args.seed_failures_2026 else seed_2026(), ensure_ascii=False))

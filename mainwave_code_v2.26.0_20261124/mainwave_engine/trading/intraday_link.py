"""Intraday observations joined to the same opportunity chain. Provisional, never a close."""
from datetime import datetime
import json
from pathlib import Path
from . import modes, runner, store

VERSION = "intraday_link_v1_20260920"
INTRADAY_ROOT = Path("results/intraday")
NOTICE = ("盘中为暂定观察：不推进生产生命周期、不改变收盘排名、不生成成交。"
          "强势确认与尾盘隔夜按数据能力保持观察态。")
STAGE = "INTRADAY_1430"
# The two sides speak different vocabularies; only a shared concern can be compared at all.
CLOSE_CONCERN = {"RISK_WATCH": "RISK", "DATA_WAIT": "UNCOMPARABLE", "MARKET_WAIT": "WAIT",
                 "WAIT_SETUP": "WAIT", "PLAN_REVIEW": "REVIEW"}
INTRADAY_CONCERN = {"SELL_REVIEW": "RISK", "REDUCE_REVIEW": "RISK", "WAIT_CLOSE_CONFIRMATION": "WAIT",
                    "HOLD_OBSERVATION": "REVIEW", "T1_LOCKED": "WAIT", "DATA_WAIT": "UNCOMPARABLE"}


def _latest(as_of, root=INTRADAY_ROOT):
    pointer = Path(root) / as_of.replace("-", "") / "latest.json"
    if not pointer.exists():
        return None
    payload = json.loads(pointer.read_text(encoding="utf-8-sig"))
    review = Path(payload.get("review", ""))
    if not review.exists():
        return None
    return json.loads(review.read_text(encoding="utf-8-sig"))


def _by_symbol(report):
    rows = {}
    for record in (report or {}).get("records", []):
        rows[record.get("symbol")] = record
    return rows


def _mode_rows(record, intraday, observation, selection):
    """Each mode answers on its own evidence; one verdict never speaks for another."""
    view = modes.evaluations(record)
    trend = next(r for r in view["evaluations"] if r["sub_scenario"] == "FIRST_PULLBACK")
    rows = [{**trend, "source": "AFTER_CLOSE", "evidence": "收盘冻结的回踩确认",
             "provisional": False}]
    strength = observation.get(record["symbol"], {}) if observation else {}
    rows.append({"mode": modes.STRENGTH_CONFIRM, "sub_scenario": "", "label": "强势确认",
                 "state": ("OBSERVED_INTRADAY" if strength else "OBSERVATION_ONLY_DATA_INSUFFICIENT"),
                 "source": "INTRADAY_1430", "evidence": strength.get("label") or "无分钟或盘口证据",
                 "executable": False, "provisional": True,
                 "confirm": modes.OBSERVATION_ONLY[modes.STRENGTH_CONFIRM]["requires"],
                 "invalidate": "收盘前封板状态可变；当前处于涨停价不等于封板成功"})
    picked = selection.get(record["symbol"], {}) if selection else {}
    rows.append({"mode": modes.OVERNIGHT, "sub_scenario": "", "label": "尾盘隔夜",
                 "state": ("OBSERVED_INTRADAY" if picked else "OBSERVATION_ONLY_NEEDS_INTRADAY_SNAPSHOT"),
                 "source": "INTRADAY_1430", "evidence": picked.get("reason") or "未进入尾盘筛选",
                 "executable": False, "provisional": True,
                 "confirm": modes.OBSERVATION_ONLY[modes.OVERNIGHT]["requires"],
                 "invalidate": "须绑定次日退出窗口；14:30后快照过期即作废"})
    return view["fields"], rows


def link(as_of, db=store.DB_PATH, root=INTRADAY_ROOT, daily_root=None):
    """Join the latest intraday review to the frozen closing snapshot, by opportunity."""
    daily_root = Path(daily_root or store.ROOT / "results/mainwave_engine")
    report = _latest(as_of, root)
    with store.connect(db) as conn:
        snapshot = store.latest(conn, as_of)
    if snapshot is None:
        return {"version": VERSION, "status": "NO_CLOSING_SNAPSHOT", "as_of": as_of,
                "notice": "没有可对照的收盘快照，盘中观察不单独成立。"}
    intraday = _by_symbol(report)
    observation = {r.get("symbol"): r for r in ((report or {}).get("tail_observation") or {}).get("records", [])}
    selection = {r.get("symbol"): r for r in ((report or {}).get("tail_selection") or {}).get("records", [])}
    rows, differences = [], []
    for record in snapshot["records"]:
        live = intraday.get(record["symbol"])
        fields, mode_rows = _mode_rows(record, live, observation, selection)
        entry = {"opportunity_id": record["opportunity_id"], "symbol": record["symbol"],
                 "name": record.get("name"), "cycle_id": record.get("cycle_id"),
                 "fields": fields, "modes": mode_rows,
                 "close_state": record.get("state"), "close_rank": record.get("strength_rank"),
                 "intraday_quote": ((live or {}).get("quote") or {}).get("close"),
                 "intraday_quality": ((live or {}).get("quote") or {}).get("quality") or "NO_QUOTE",
                 "provisional": True}
        close_concern = CLOSE_CONCERN.get(record.get("state"), "UNCOMPARABLE")
        live_concern = INTRADAY_CONCERN.get((live or {}).get("status"), "UNCOMPARABLE")
        entry["comparable"] = (entry["intraday_quality"] != "NO_QUOTE"
                               and "UNCOMPARABLE" not in (close_concern, live_concern))
        if not entry["comparable"]:
            entry["difference_note"] = ("盘中无新鲜报价或判断口径不可比，不计为差异"
                                        if live else "当日盘中未覆盖该标的")
        elif close_concern != live_concern:
            entry["difference"] = {"close": record.get("state"), "close_concern": close_concern,
                                   "intraday": live["status"], "intraday_concern": live_concern,
                                   "reading": "盘中暂定判断与收盘冻结判断不一致，以收盘为准，差异留痕"}
            differences.append(entry["difference"] | {"symbol": record["symbol"]})
        rows.append(entry)
    observed = [r for r in rows if r["intraday_quality"] != "NO_QUOTE"]
    return {"version": VERSION, "status": "READY" if report else "NO_INTRADAY_RUN", "as_of": as_of,
            "observed_at": (report or {}).get("observed_at", ""),
            "market": (report or {}).get("market", {}),
            "records": rows, "differences": differences,
            "summary": {"opportunities": len(rows), "with_intraday_quote": len(observed),
                        "comparable": sum(1 for r in rows if r["comparable"]),
                        "strength_observed": sum(1 for r in rows for m in r["modes"]
                                                 if m["mode"] == modes.STRENGTH_CONFIRM and m["state"] == "OBSERVED_INTRADAY"),
                        "overnight_observed": sum(1 for r in rows for m in r["modes"]
                                                  if m["mode"] == modes.OVERNIGHT and m["state"] == "OBSERVED_INTRADAY"),
                        "differences": len(differences)},
            "notice": NOTICE}


def session(as_of, db=store.DB_PATH, root=INTRADAY_ROOT, at=None):
    """Repeatable 14:30 pass: one lock, one checkpoint per attempt, risk before new ideas."""
    name = f"intraday:{as_of}"
    lock = runner.acquire(name, db, at=at)
    if not lock["acquired"]:
        return {"version": VERSION, "status": "ANOTHER_RUN_IN_PROGRESS", "as_of": as_of, "lock": lock,
                "notice": "重复调度不重复评估，也不重复创建任何模拟成交。"}
    try:
        pending = runner.pending(as_of, db)
        freshness = runner.freshness(as_of)
        joined = link(as_of, db, root)
        runner.checkpoint(as_of, STAGE, "link", "DONE",
                          {"status": joined["status"], "records": len(joined.get("records", []))}, db, at=at)
        return {"version": VERSION, "status": joined["status"], "as_of": as_of, "lock": lock,
                "freshness": freshness, "risk_first": pending["order"],
                "pending_tasks": pending["tasks"], "link": joined,
                "notice": "先处理退出与持仓风险，再看新增机会；盘中结果为暂定，收盘由正式行情接管。"}
    finally:
        runner.release(name, db)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="把盘中暂定观察并入收盘机会链（只读，不下单）")
    parser.add_argument("--date", required=True)
    args = parser.parse_args()
    day = args.date if "-" in args.date else f"{args.date[:4]}-{args.date[4:6]}-{args.date[6:]}"
    result = session(day)
    output = Path(store.ROOT) / "results/mainwave_engine" / day.replace("-", "")
    output.mkdir(parents=True, exist_ok=True)
    (output / "intraday_link.json").write_text(store.encoded(result), encoding="utf-8")
    # Several intraday runs a day: the latest stays at intraday_link.json, every run is also kept on its own.
    runs = output / "intraday_link_runs"
    runs.mkdir(parents=True, exist_ok=True)
    (runs / f"{datetime.now().strftime('%H%M%S')}.json").write_text(store.encoded(result), encoding="utf-8")
    summary = result.get("link", {}).get("summary", {})
    print(f"[intraday_link] {day} status={result['status']} "
          f"comparable={summary.get('comparable', 0)} differences={summary.get('differences', 0)} "
          f"-> {output / 'intraday_link.json'}", flush=True)
    return 0 if result["status"] in {"READY", "NO_INTRADAY_RUN"} else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

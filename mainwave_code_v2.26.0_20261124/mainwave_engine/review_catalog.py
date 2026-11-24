"""Complete frozen R1 research cohort; full case evidence is materialized on demand."""
from __future__ import annotations

from datetime import date
from functools import lru_cache
import json
import math
from pathlib import Path
from threading import Lock

import pandas as pd

from . import review_workbench as RW

POLICIES = {"legacy20", "stage36_20", "structure36_20"}
POLICY_ORDER = ("legacy20", "stage36_20", "structure36_20")
SELECTION_NOTICE = "同一信号仅保留已平仓方案中净收益最高的一例；这是使用未来结果的事后择优，不是可执行策略、胜率或OOS表现。未结束/待核验方案不按零收益参与比较。"
MATERIALIZE_LOCK = Lock()


def case_id(row):
    return f"R1_{row['symbol']}_{row['date'].replace('-', '')}_{row['exit']}"


def build(root=RW.ROOT, source=RW.SOURCE):
    if (root / "catalog.json").exists():
        return load(root)
    source_bytes = (source / "r1_exit_replay.csv.gz").read_bytes()
    frame = pd.read_csv(source / "r1_exit_replay.csv.gz")
    frame = frame[frame.year.eq(2026)].sort_values(["date", "symbol", "exit"])
    if frame.empty or not set(frame.exit).issubset(POLICIES) or frame.duplicated(["symbol", "date", "exit"]).any():
        raise ValueError("年度回测母集为空、退出规则未知或案例身份重复")
    manifest = json.loads((source / "exit_replay_manifest.json").read_text(encoding="utf-8"))
    mapping = RW.SA._read_json(Path("concept_mapping.json"), {}).get("mapping", {})
    frozen = root / "catalog_source"
    (frozen / "data").mkdir(parents=True, exist_ok=True)
    from research.route_clock import sessions
    inputs = {r["symbol"]: r["price_hash"] for r in manifest["inputs"]}
    errors = {}
    # Freeze only this cohort's inputs, so later daily updates cannot change case evidence.
    for symbol in frame.symbol.unique():
        target = frozen / "data" / f"{symbol}_daily.csv"
        try:
            data = target.read_bytes() if target.exists() else (Path("data") / target.name).read_bytes()
            if RW.digest(data) != inputs.get(symbol):
                raise ValueError("行情哈希与原回测不一致")
            if not target.exists():
                target.write_bytes(data)
        except (OSError, ValueError) as exc:
            errors[symbol] = str(exc)
    calendar = [d for d in sessions() if d <= manifest["as_of"]]
    manifest = {**manifest, "frozen_calendar": calendar}
    (frozen / "exit_replay_manifest.json").write_bytes(RW.encoded(manifest))
    (frozen / "r1_exit_replay.csv.gz").write_bytes(source_bytes)
    archived = {p.stem: RW.detail(p.stem, root)["summary"] for p in (root / "cases").glob("*.json")}
    entries = {}
    for row in RW.records(frame):
        cid = case_id(row)
        summary = {"id": cid, "symbol": row["symbol"], "name": mapping.get(row["symbol"], {}).get("name", row["symbol"]),
                   "route": "R1", "signal_date": row["date"], "exit_policy": row["exit"],
                   "trade_status": row["status"], "net_pct": row.get("net"), "window_end": manifest["as_of"],
                   "input_error": errors.get(row["symbol"], "")}
        summary.update(archived.get(cid, {}))
        entries[cid] = {"summary": summary, "source_row": row, "row_hash": RW.digest(RW.encoded(row))}
    result = {"version": "review_catalog_v1", "year": 2026, "as_of": manifest["as_of"],
              "source_hash": RW.digest(source_bytes), "manifest_hash": RW.digest(RW.encoded(manifest)),
              "signals": len(frame.drop_duplicates(["symbol", "date"])), "entries": entries,
              "scope": "2026年R1固定母集全量，含3种退出分支；R2覆盖研究、R3传播代理不混入交易母集。不是组合收益或未见OOS。"}
    result["catalog_hash"] = RW.digest(RW.encoded(result))
    with (root / "catalog.json").open("xb") as handle:
        handle.write(RW.encoded(result))
    return result


@lru_cache(maxsize=2)
def _load(path, mtime, size):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value["catalog_hash"] != RW.digest(RW.encoded({k: v for k, v in value.items() if k != "catalog_hash"})):
        raise ValueError("年度案例目录校验失败")
    return value


def load(root=RW.ROOT):
    path = root / "catalog.json"
    stat = path.stat()
    return _load(str(path), stat.st_mtime_ns, stat.st_size)


def materialize(cid, root=RW.ROOT):
    catalog = load(root)
    item = catalog["entries"].get(cid)
    if item is None:
        raise FileNotFoundError("不在冻结年度案例清单中")
    if item["summary"].get("input_error"):
        raise ValueError(item["summary"]["input_error"])
    source = root / "catalog_source"
    with MATERIALIZE_LOCK:
        if (root / "cases" / f"{cid}.json").exists():
            return
        if RW.digest((source / "r1_exit_replay.csv.gz").read_bytes()) != catalog["source_hash"] or RW.digest((source / "exit_replay_manifest.json").read_bytes()) != catalog["manifest_hash"]:
            raise ValueError("冻结回测输入校验失败")
        RW.import_r1_case(item["source_row"], "2026年R1固定信号母集全量导入，保留全部退出分支及未决/未成交记录；未按收益选择。", root, source)


def best_per_signal(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["route"], row["symbol"], row["signal_date"]), []).append(row)
    selected = []
    for group in groups.values():
        ordered = sorted(group, key=lambda r: POLICY_ORDER.index(r["exit_policy"]))
        closed = [r for r in ordered if r["trade_status"] == "CLOSED" and
                  r.get("net_pct") is not None and math.isfinite(r["net_pct"]) and not r.get("input_error")]
        pending = [r for r in ordered if r not in closed and
                   not r["trade_status"].startswith(("ENTRY", "INVALID"))]
        if closed:
            winner = max(closed, key=lambda r: round(r["net_pct"], 10))
            label = "已平仓中暂优，仍有方案待定" if pending else "事后最优"
        else:
            winner = (pending or ordered)[0]
            label = "待定，暂留一例" if pending else "未成交，暂留一例"
        selection = {"version": "best_closed_net_v1", "selected_id": winner["id"], "label": label,
                     "notice": SELECTION_NOTICE, "closed_count": len(closed), "pending_count": len(pending),
                     "tie_break": "净收益保留10位小数相同，依次保留legacy20、stage36_20、structure36_20；不使用人工结论选优。",
                     "alternatives": [{k: r.get(k) for k in ("id", "exit_policy", "trade_status", "net_pct")} for r in ordered],
                     "related_reviews": [{k: r.get(k) for k in ("id", "exit_policy", "revision", "review_status")}
                                         for r in ordered if r["id"] != winner["id"] and r.get("revision", 0)]}
        saved = [r for r in ordered if r.get("revision", 0)]
        signal_review_status = ("REVIEWED" if any(r.get("review_status") == "REVIEWED" for r in saved)
                                else "DRAFT" if saved else "UNREVIEWED")
        selected.append({**winner, "exit_selection": selection, "signal_review_status": signal_review_status})
    return selected


def review_order(row):
    reviewed = (row.get("signal_review_status") == "REVIEWED"
                or (row.get("revision", 0) > 0 and row.get("review_status") == "REVIEWED"))
    day = date.fromisoformat(row["signal_date"]).toordinal()
    return (0 if reviewed else 1), (day if reviewed else -day), row["symbol"], row["id"]


def selection_for(cid, root=RW.ROOT):
    if not (root / "catalog.json").exists():
        return None
    for row in listing(root)["cases"]:
        selection = row["exit_selection"]
        if any(r["id"] == cid for r in selection["alternatives"]):
            return selection
    return None


def listing(root=RW.ROOT):
    catalog = load(root)
    reviews = RW.saved_reviews(root)
    archived = {p.stem for p in (root / "cases").glob("*.json")}
    rows = [{**item["summary"], "materialized": cid in archived,
             "revision": reviews.get(cid, {}).get("revision", 0),
             "review_status": reviews.get(cid, {}).get("status", "DRAFT")}
            for cid, item in catalog["entries"].items()]
    raw_count = len(rows)
    rows = best_per_signal(rows)
    rows.sort(key=review_order)
    return {"cases": rows, "notice": RW.NOTICE, "stages": RW.STAGES, "signals": catalog["signals"],
            "raw_cases": raw_count, "scope": f"2026年R1固定母集，每个信号一例。{SELECTION_NOTICE} 原始退出分支及人工分析保留归档；R2/R3不混入。",
            "catalog_hash": catalog["catalog_hash"]}

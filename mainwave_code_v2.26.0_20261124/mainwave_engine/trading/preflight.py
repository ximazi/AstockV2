"""Startup self-check and data contract. Missing input is named, never counted as zero."""
import hashlib
import json
from pathlib import Path
import shutil
import pandas as pd
from . import store

VERSION = "preflight_contract_v1_20260920"
# Local bars are unadjusted (adjustflag=3): volume in shares, amount in CNY, turnover in percent.
CONTRACT = {
    "price_basis": "RAW_UNADJUSTED", "adjust_flag": 3, "volume_unit": "SHARES",
    "amount_unit": "CNY", "turnover_unit": "PERCENT",
    "return_index": "pct_chg 复利收益指数用于比较；执行价格用原始价",
    "float_shares": "NOT_DELIVERED_LOCALLY",
}
REQUIRED = "REQUIRED"
OPTIONAL = "OPTIONAL"
PACKAGE = "knowledge_package"
REGISTER = "02_知识点覆盖与待验证.md"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _check(cid, label, tier, ok, detail, impact="", unknown=False):
    return {"id": cid, "label": label, "tier": tier, "detail": detail, "impact": impact,
            "status": "UNKNOWN" if unknown else "PASS" if ok else "FAIL"}


def manifest(root=store.ROOT):
    """Portable source manifest; a machine-specific folder is never the delivery."""
    root = Path(root)
    register = root / REGISTER
    if not register.exists():
        return {"version": VERSION, "status": "REGISTER_MISSING", "register": REGISTER, "sources": []}
    sources = [{"path": path.relative_to(root).as_posix(), "sha256": _sha(path),
                "bytes": path.stat().st_size}
               for path in sorted((root / "talk/transcripts").rglob("*.transcript.reviewed.md"))]
    return {"version": VERSION, "status": "READY" if sources else "SOURCES_MISSING",
            "register": REGISTER, "register_sha256": _sha(register),
            "generated_at": store.now(), "sources": sources}


def export_package(root=store.ROOT, dest=None):
    """Write the manifest plus referenced files so a clean checkout can run."""
    root = Path(root)
    dest = Path(dest or root / "results/trading_system" / PACKAGE)
    book = manifest(root)
    if book["status"] != "READY":
        raise ValueError(f"来源资料不完整，无法交付：{book['status']}")
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / book["register"], dest / Path(book["register"]).name)
    for item in book["sources"]:
        target = dest / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / item["path"], target)
    (dest / "manifest.json").write_text(store.encoded(book), encoding="utf-8")
    return {**book, "package": str(dest)}


def import_package(source, root=store.ROOT):
    """Install a delivered package only when every hash matches the manifest."""
    source, root = Path(source), Path(root)
    book = json.loads((source / "manifest.json").read_text(encoding="utf-8-sig"))
    register = source / Path(book["register"]).name
    if _sha(register) != book["register_sha256"]:
        raise ValueError("登记册哈希不匹配，拒绝导入")
    for item in book["sources"]:
        if _sha(source / item["path"]) != item["sha256"]:
            raise ValueError(f"来源哈希不匹配，拒绝导入：{item['path']}")
    shutil.copyfile(register, root / book["register"])
    for item in book["sources"]:
        target = root / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / item["path"], target)
    return {"version": VERSION, "imported": len(book["sources"]), "register": book["register"]}


def _calendar(data_root, day):
    path = data_root / "000300_SH_index.csv"
    if not path.exists():
        return None, _check("calendar", "交易日历与基准指数", REQUIRED, False,
                            "缺少 000300_SH_index.csv", "无法判定交易日、缺日与基准超额")
    try:
        dates = pd.read_csv(path, usecols=["trade_date"], dtype=str).trade_date
    except (OSError, ValueError) as exc:
        return None, _check("calendar", "交易日历与基准指数", REQUIRED, False,
                            f"无法读取 {path.name}：{exc}", "不能用其他日期代替目标日")
    parsed = pd.to_datetime(dates.str.strip().str.replace("-", "", regex=False), format="%Y%m%d", errors="coerce")
    days = sorted(set(parsed.dropna().dt.strftime("%Y-%m-%d")))
    if day not in days:
        return days, _check("calendar", "交易日历与基准指数", REQUIRED, False,
                            f"{path.name} 缺少 {day}，本地最新有效日期：{days[-1] if days else '无'}",
                            "该日不按交易日处理，不沿用前一日冒充")
    return days, _check("calendar", "交易日历与基准指数", REQUIRED, True,
                        f"覆盖 {days[0]}~{days[-1]}，共 {len(days)} 个交易日")


def _contract_check(data_root, symbol, day):
    path = data_root / f"{symbol}_daily.csv"
    if not path.exists():
        return _check("contract", "行情单位与复权基准", REQUIRED, False, f"缺少样本行情 {symbol}",
                      "无法核对成交量、成交额、换手与复权基准")
    frame = pd.read_csv(path).tail(20)
    flags = set(pd.to_numeric(frame.adjustflag, errors="coerce").dropna().astype(int))
    row = frame.iloc[-1]
    amount, volume, close = float(row.amount), float(row.volume), float(row.close)
    consistent = volume > 0 and abs(amount / (volume * close) - 1) < .2
    turnover = pd.to_numeric(frame.turnover, errors="coerce").dropna()
    percent = bool(len(turnover)) and float(turnover.max()) <= 100
    ok = flags == {CONTRACT["adjust_flag"]} and consistent and percent
    return _check("contract", "行情单位与复权基准", REQUIRED, ok,
                  f"adjustflag={sorted(flags)}；amount≈volume×close：{consistent}；换手为百分比：{percent}",
                  "" if ok else "单位或复权基准与契约不符，价格与成交口径需人工核对")


def _bars(data_root, symbols, day, days):
    """Stale, missing and mid-series gaps are reported apart; none of them is a zero."""
    missing, stale, gapped, ready = [], [], [], 0
    window = [d for d in (days or []) if d <= day][-20:]
    for symbol in symbols:
        path = data_root / f"{symbol}_daily.csv"
        try:
            dates = set(pd.read_csv(path, usecols=["date"]).date.astype(str).str[:10])
        except (OSError, ValueError, KeyError):
            missing.append(symbol)
            continue
        if day not in dates:
            stale.append(symbol)
        elif window and [d for d in window if d not in dates]:
            gapped.append(symbol)
        else:
            ready += 1
    return missing, stale, gapped, ready


def check(day, root=store.ROOT, db=store.DB_PATH, data_root=None, sample_symbols=None):
    root = Path(root)
    data_root = Path(data_root or root / "data")
    output = root / "results/mainwave_engine" / day.replace("-", "")
    checks = []

    days, calendar_check = _calendar(data_root, day)
    checks.append(calendar_check)

    universe = evidence = None
    for name, key in (("analysis_universe.json", "universe"), ("analysis_evidence.json", "evidence")):
        path = output / name
        if not path.exists():
            checks.append(_check(key, f"{name} 当日分析结果", REQUIRED, False, "文件缺失",
                                 "无法构建候选域与证据，闭环不应继续"))
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except ValueError as exc:
            checks.append(_check(key, f"{name} 当日分析结果", REQUIRED, False, f"解析失败：{exc}", "输入不可信"))
            continue
        same = payload.get("as_of") == day
        checks.append(_check(key, f"{name} 当日分析结果", REQUIRED, same,
                             f"as_of={payload.get('as_of')}，记录 {len(payload.get('records') or [])} 条",
                             "" if same else "日期不匹配，不能用旧结果冒充本次成功"))
        if key == "universe":
            universe = payload
        else:
            evidence = payload

    symbols = sample_symbols if sample_symbols is not None else [
        r["symbol"] for r in (universe or {}).get("records", [])]
    missing, stale, gapped, ready = _bars(data_root, symbols, day, days)
    affected = len(missing) + len(stale) + len(gapped)
    # A usable universe is required; individual halts or gaps only disqualify those symbols.
    checks.append(_check("bars", "候选行情可用域", REQUIRED, bool(symbols) and ready > 0,
                         f"可用 {ready}/{len(symbols)}",
                         "" if ready else "没有任何候选具备当日行情，闭环不应继续", unknown=not symbols))
    checks.append(_check("bars_coverage", "逐股行情完整性", OPTIONAL, not affected,
                         f"受影响 {affected}：缺失 {len(missing)}，未更新 {len(stale)}，中间缺日 {len(gapped)}",
                         "受影响个股按数据待核处理，不参与排名，也不记为零；"
                         "缺日可能是停牌，也可能是抓取遗漏，不单独作为停牌证据",
                         unknown=not symbols))
    checks.append(_contract_check(data_root, (symbols or ["sh_600000"])[0], day))

    try:
        with store.connect(db) as conn:
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        wanted = {"snapshots", "opportunities", "opportunity_events", "registry_versions", "assessments"}
        checks.append(_check("database", "记录库结构", REQUIRED, wanted <= tables,
                             f"已建表 {len(tables & wanted)}/{len(wanted)}",
                             "" if wanted <= tables else "缺表将导致连续性记录丢失"))
    except Exception as exc:  # sqlite errors carry the actionable reason
        checks.append(_check("database", "记录库结构", REQUIRED, False, str(exc), "无法写入连续性记录"))

    book = manifest(root)
    checks.append(_check("sources", "知识来源交付", OPTIONAL, book["status"] == "READY",
                         f"状态 {book['status']}，来源 {len(book['sources'])} 份",
                         "" if book["status"] == "READY" else "知识引用与规则溯源降级，其余分析可继续"))

    records = (universe or {}).get("records", [])
    themed = sum(1 for r in records if r.get("belonging_theme"))
    checks.append(_check("themes", "题材归属覆盖", OPTIONAL, bool(records) and themed / len(records) >= .5,
                         f"{themed}/{len(records)} 条候选有题材归属",
                         "题材比较域收窄，同题材相对强度按不足样本处理", unknown=not records))

    leader = output / "leader_state_daily.csv"
    checks.append(_check("leaders", "日级领头线索", OPTIONAL, leader.exists(),
                         "已生成" if leader.exists() else "缺少 leader_state_daily.csv",
                         "角色判定退回未确认，不影响强势排名"))

    required_failed = [c for c in checks if c["tier"] == REQUIRED and c["status"] != "PASS"]
    optional_failed = [c for c in checks if c["tier"] == OPTIONAL and c["status"] != "PASS"]
    return {"version": VERSION, "as_of": day, "contract": CONTRACT, "checks": checks,
            "status": "BLOCKED" if required_failed else "DEGRADED" if optional_failed else "READY",
            "blocking": [c["id"] for c in required_failed], "degraded": [c["id"] for c in optional_failed],
            "data_wait": {"missing": missing[:50], "stale": stale[:50], "gapped": gapped[:50],
                          "counts": {"missing": len(missing), "stale": len(stale),
                                     "gapped": len(gapped), "ready": ready}},
            "notice": "必要项失败应停止闭环；辅助项失败记为降级并保留原因，缺失不以0计。"}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Check the required benchmark calendar before the daily scan.")
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    day = parser.parse_args().date
    _, calendar = _calendar(store.ROOT / "data", day)
    print(f"[calendar_preflight] {calendar['status']} | {calendar['detail']}", flush=True)
    if calendar["status"] != "PASS":
        print(f"[calendar_preflight] 核对是否交易日。若是，先补指数："
              f"python update_a_share_major_indices.py --end-date {day.replace('-', '')}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

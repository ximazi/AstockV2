"""Send the bounded evidence packet to a model and merge its answer as a new report version.

The model only ever sees ai_packet.json (facts, gaps, mode contracts) under the v2 analyzer prompt; its answer
is data: parsed, checked against the prompt's output contract, unknown symbols and source ids dropped, then
merged into the explanation layer. Facts, prices, statuses and the rule verdict are never written by a model.
Provider is configuration (.env): FUPAN_ANALYZER = codex-cli | off (default: local Codex CLI).
Calls are batched and cached by facts, model, prompt and transport version."""
import json
import time
from datetime import datetime
from pathlib import Path

from .. import eod_review as R
from .. import model_cli
from . import ASSETS, VERSION, AI_SCHEMA_VERSION
from .build import TZ, apply_analysis, latest_report

RESULTS = ("SUPPORTED", "UNSUPPORTED", "NOT_APPLICABLE", "UNKNOWN")
SECTION_IDS = ("01", "02", "03", "04", "05", "06", "07")


def config(env=None):
    return model_cli.config(env)


def system_prompt():
    return (ASSETS / "prompt_analyzer_v2.md").read_text(encoding="utf-8")


BATCH_SIZE = 8  # stocks per model call; a 20+ stock answer overflows one CLI message and comes back truncated


def batches(packet):
    symbols = [s["symbol"] for s in packet.get("stocks", [])]
    return [symbols[i:i + BATCH_SIZE] for i in range(0, len(symbols), BATCH_SIZE)] or [[]]


def user_prompt(packet, symbols=None, first=True):
    """The full packet every time (comparisons need all of it); a batch only narrows what is written."""
    if symbols is None:
        scope = "section_assessments 恰好覆盖 01–07 各一次；stock_assessments 覆盖输入的每只股票且只写输入内的证券；"
    else:
        scope = ("本次为分批评估：stock_assessments 只写这 " + str(len(symbols)) + " 只：" + "、".join(symbols)
                 + "（其他股票只作对比依据，不写入）；"
                 + ("section_assessments 恰好覆盖 01–07 各一次，按全部输入股票与市场事实来写，不要写成“本批”；" if first
                    else "section_assessments 输出空数组 []；"))
    return ("## 输入事实包（review.ai.input.v2）\n\n```json\n" + json.dumps(packet, ensure_ascii=False, indent=1)
            + "\n```\n\n只输出提示词规定的 JSON 对象（as_of 必须等于 time_boundary.trade_date 即 "
            + str((packet.get("time_boundary") or {}).get("trade_date", "")) + "；objective_confirmation；coverage；"
            + scope +
            "所有 support_ids/contrary_ids 只能引用 source_catalog 中的 id；mode_assessments.result 只能是 "
            + "/".join(RESULTS) + "；pairwise_comparisons、complete_plan_ids、bounded_data_requests、unresolved_conflicts），"
            "不输出 HTML，不输出 JSON 以外的文字。\n")


def merge_batches(parts, groups, packet):
    """One answer from batch answers: sections from the first, each batch contributes only its own stocks."""
    merged = dict(parts[0])
    merged["stock_assessments"] = [a for part, group in zip(parts, groups)
                                   for a in (part.get("stock_assessments") or []) if isinstance(a, dict) and a.get("symbol") in group]
    for key in ("pairwise_comparisons", "complete_plan_ids", "bounded_data_requests", "unresolved_conflicts"):
        items, ids = [], set()
        for n, part in enumerate(parts):
            for item in part.get(key) or []:
                if isinstance(item, dict) and item.get("request_id"):
                    if item["request_id"] in ids:
                        item = {**item, "request_id": f"{item['request_id']}_B{n + 1}"}
                    ids.add(item["request_id"])
                if item not in items:
                    items.append(item)
        merged[key] = items
    covered = [a.get("symbol") for a in merged["stock_assessments"]]
    required = list(packet.get("required_symbols", []))
    merged["coverage"] = {"required_symbols": required, "covered_symbols": covered,
                          "missing_symbols": [s for s in required if s not in covered]}
    return merged


TRANSPORTS = {"codex-cli": model_cli.call}


# ---------------------------------------------------------------- answer handling
def _objects(text):
    """Every top-level JSON object in text, in order."""
    decoder, found, i = json.JSONDecoder(), [], text.find("{")
    while i >= 0:
        try:
            obj, end = decoder.raw_decode(text, i)
        except ValueError:
            i = text.find("{", i + 1)
            continue
        if isinstance(obj, dict):
            found.append(obj)
        i = text.find("{", end)
    return found


def parse(text):
    """The answer object in the model's text; fences, chatter and any extra JSON objects around it are ignored.
    The object carrying section_assessments wins, else the largest one."""
    body = text.strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[1] if "\n" in body else body[3:]
        body = body.rsplit("```", 1)[0]
    found = _objects(body)
    if not found:
        raise ValueError("模型没有返回 JSON 对象")
    answers = [o for o in found if "section_assessments" in o]
    return answers[0] if answers else max(found, key=lambda o: len(json.dumps(o, ensure_ascii=False)))


CHECK_VERSION = "analysis_contract_v2_20260923"


def check(analysis, packet):
    """The output contract, every item a refusal: seven sections exactly once, only packet securities, every
    required security assessed, enum whitelist, evidence ids from the packet's catalog, as_of equal to the packet's
    trade date. A model answer that fails any of these is data the report must not absorb."""
    problems = []
    if not isinstance(analysis, dict):
        return ["返回不是 JSON 对象"]
    sections = analysis.get("section_assessments")
    if not isinstance(sections, list) or not sections:
        problems.append("缺 section_assessments")
    else:
        ids = [str(a.get("section_id", "")).zfill(2) for a in sections if isinstance(a, dict)]
        missing = [sid for sid in SECTION_IDS if sid not in ids]
        dup = sorted({sid for sid in ids if ids.count(sid) > 1})
        if missing:
            problems.append("section_assessments 未覆盖 " + "、".join(missing))
        if dup:
            problems.append("section_assessments 重复 " + "、".join(dup))
        unknown = sorted({sid for sid in ids if sid not in SECTION_IDS})
        if unknown:
            problems.append("section_assessments 含未知章节 " + "、".join(unknown))
    symbols = {s["symbol"] for s in packet.get("stocks", [])}
    catalog = {s["id"] for s in packet.get("source_catalog", [])}
    stocks = analysis.get("stock_assessments")
    if not isinstance(stocks, list):
        problems.append("缺 stock_assessments")
    else:
        outside = sorted({str(a.get("symbol")) for a in stocks if isinstance(a, dict) and a.get("symbol") not in symbols})
        if outside:
            problems.append("stock_assessments 含输入外的证券：" + "、".join(outside[:5]))
        bad = sorted({str(m.get("result")) for a in stocks if isinstance(a, dict) for m in a.get("mode_assessments") or [] if isinstance(m, dict) and m.get("result") not in RESULTS})
        if bad:
            problems.append("mode_assessments.result 非法值：" + "、".join(bad))
        required = set(packet.get("required_symbols", []))
        covered = {a.get("symbol") for a in stocks if isinstance(a, dict)}
        if required - covered:
            problems.append("必保留股未评估：" + "、".join(sorted(required - covered)))
    cited = set()
    for a in (sections if isinstance(sections, list) else []) + (stocks if isinstance(stocks, list) else []):
        if not isinstance(a, dict):
            continue
        for key in ("support_ids", "contrary_ids"):
            cited.update(str(x) for x in (a.get(key) or []) if isinstance(x, str))
        for m in a.get("mode_assessments") or []:
            if isinstance(m, dict):
                cited.update(str(x) for x in (m.get("support_ids") or []) if isinstance(x, str))
    unknown_ids = sorted(cited - catalog) if catalog else []
    if unknown_ids:
        problems.append("引用了事实包之外的证据ID：" + "、".join(unknown_ids[:5]))
    trade_date = str((packet.get("time_boundary") or {}).get("trade_date", ""))
    as_of = analysis.get("as_of")
    if not as_of:
        problems.append("缺 as_of")
    elif str(as_of) != trade_date:
        problems.append(f"as_of 与输入交易日不一致：{as_of} ≠ {trade_date}")
    return problems


def blocking(problems):
    """Every contract problem blocks; there is no warn-and-merge path."""
    return list(problems)


# ---------------------------------------------------------------- run
def run(day, out_root=R.OUT_ROOT, cfg=None, client=None, force=False):
    """Analyse the day's latest report: reuse an unchanged answer, else call bounded batches and merge."""
    cfg = cfg or config()
    folder = Path(out_root) / day.replace("-", "") / "fupan"
    current = latest_report(out_root, day)
    if not current:
        return {"status": "NO_REPORT", "detail": f"{day} 没有复盘报告"}
    version_dir = Path(current["_path"]).parent
    pointer = R.read_json(folder / "latest.json")
    if pointer.get("rule_path"):
        version_dir = Path(pointer["rule_path"])
    packet = R.read_json(version_dir / "ai_packet.json")
    stats = R.read_json(version_dir / "packet_stats.json")
    sha = stats.get("ai_packet_sha256") or R.digest(packet)
    # Cached by the facts behind the report (base_hash) plus every version that shapes the answer: provider/model,
    # prompt text, packet schema, contract check. A cached answer is re-validated against the current contract.
    prompt_sha = R.digest((ASSETS / "prompt_analyzer_v2.md").read_text(encoding="utf-8"))[:12]
    key = R.digest([pointer.get("base_hash") or sha, cfg["provider"], cfg["model"], prompt_sha,
                    AI_SCHEMA_VERSION, CHECK_VERSION, model_cli.VERSION])[:16]
    if pointer.get("analysis_merged") and not force:
        meta = R.read_json(Path(current["_path"]).parent / "analysis_meta.json")
        if meta.get("cache_key") == key:
            return {"status": "ALREADY_MERGED", **meta, "pointer": pointer}
    cache = folder / "analysis" / f"{key}.json"
    meta_path = cache.with_suffix(".meta.json")
    if cache.exists() and not force:
        analysis, meta = R.read_json(cache), R.read_json(meta_path)
        meta["cached"] = True
        meta["problems"] = check(analysis, packet)
        if blocking(meta["problems"]):
            return {"status": "REJECTED_CACHED", **meta}
    else:
        transport = client or TRANSPORTS.get(cfg["provider"])
        if transport is None:
            return {"status": "SKIPPED_NO_PROVIDER", "provider": cfg["provider"],
                    "detail": "分析器已关闭；默认使用本地 codex-cli，需要已安装并执行 codex login"}
        started = time.perf_counter()
        groups = batches(packet)
        parts, texts, usage = [], [], {}
        for n, group in enumerate(groups):
            label = f"第 {n + 1}/{len(groups)} 批：" if len(groups) > 1 else ""
            try:
                text, used = transport(system_prompt(), user_prompt(packet, group if len(groups) > 1 else None, n == 0), cfg)
            except Exception as exc:  # the report stays the rule version; the failure is recorded, never hidden
                return {"status": "FAILED", "provider": cfg["provider"], "model": cfg["model"], "error": (label + str(exc))[:500],
                        "batches": len(groups), "latency_s": round(time.perf_counter() - started, 1)}
            try:
                parts.append(parse(text))
            except Exception as exc:  # a paid answer that cannot be read is kept for diagnosis, not discarded
                (folder / "analysis").mkdir(parents=True, exist_ok=True)
                raw = folder / "analysis" / f"{sha[:16]}.b{n + 1}.unparsed.txt"
                raw.write_text(text, encoding="utf-8")
                return {"status": "FAILED", "provider": cfg["provider"], "model": cfg["model"], "error": f"PARSE {label}{str(exc)[:400]}",
                        "raw_answer": str(raw), "batches": len(groups), "latency_s": round(time.perf_counter() - started, 1)}
            texts.append(text)
            for k, v in (used or {}).items():
                usage[k] = usage.get(k, 0) + v if isinstance(v, (int, float)) and not isinstance(v, bool) else v
        analysis = merge_batches(parts, groups, packet) if len(groups) > 1 else parts[0]
        # Host provenance participates in the merge hash, even if two providers return identical wording.
        analysis["_local_origin"] = {"provider": cfg["provider"], "model": cfg["model"], "cache_key": key}
        text = "\n\n".join(texts)
        problems = check(analysis, packet)
        meta = {"provider": cfg["provider"], "model": cfg["model"], "prompt_version": VERSION, "prompt_sha": prompt_sha, "check_version": CHECK_VERSION,
                "cache_key": key, "transport_version": model_cli.VERSION,
                "schema_version": AI_SCHEMA_VERSION, "packet_sha256": sha,
                "called_at": datetime.now(TZ).isoformat(timespec="seconds"), "latency_s": round(time.perf_counter() - started, 1),
                "usage": usage, "problems": problems, "cached": False, "raw_chars": len(text), "batches": len(groups)}
        if blocking(problems):
            (folder / "analysis").mkdir(parents=True, exist_ok=True)
            (folder / "analysis" / f"{sha[:16]}.rejected.txt").write_text(text, encoding="utf-8")
            return {"status": "REJECTED", **meta}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(analysis, ensure_ascii=False, indent=1), encoding="utf-8")
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    try:
        pointer = apply_analysis(day, analysis, out_root)
    except ValueError as exc:
        return {"status": "MERGE_FAILED", **meta, "error": str(exc)[:500]}
    report_meta = Path(pointer["path"]) / "analysis_meta.json"
    if not report_meta.exists():
        report_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"status": "MERGED", **meta, "pointer": pointer}

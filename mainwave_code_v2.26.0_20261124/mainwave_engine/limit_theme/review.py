"""Full limit-up coverage with resumable Codex evidence reviews; never writes membership or ledgers."""
import copy
import html
import json
import re
import time
from collections import Counter
from pathlib import Path

from .. import eod_review as R, intraday_quotes as Q, model_cli
from . import VERSION
from .sources import Collector, text

BATCH = 6
STATUSES = {"CORROBORATED": "交叉支持", "DISPUTED": "存在分歧", "INSUFFICIENT": "证据不足"}
PROMPT = Path(__file__).with_name("prompt.md")


def unknown(packet, reason, processing="INSUFFICIENT_MATERIAL"):
    return {"symbol": packet["symbol"], "primary_theme": "待核", "status": "INSUFFICIENT", "hypotheses": [],
            "business_reality": {"status": "UNKNOWN", "summary": "未确认", "citations": []},
            "lhb_reading": {"summary": "以原始披露及采集状态为准", "citations": []},
            "processing": processing, "unknowns": [reason], "next_checks": ["补齐或复核原始材料"]}


def check(answer, packet):
    """Only cited original material can corroborate an attribution; legacy tags and LHB cannot."""
    if (not isinstance(answer, dict) or answer.get("symbol") != packet["symbol"] or answer.get("status") not in STATUSES
            or not isinstance(answer.get("primary_theme"), str) or not 1 <= len(answer["primary_theme"].strip()) <= 40
            or not isinstance(answer.get("hypotheses"), list)):
        raise ValueError("Invalid stock attribution contract")
    if set(answer) & {"target_price", "win_rate", "probability", "expected_return", "orders", "buy_qty"}:
        raise ValueError("Trading authority/forecast fields are forbidden")
    known = {e["id"]: e for e in packet["materials"]}

    def citations(items):
        if not isinstance(items, list):
            raise ValueError("Citations must be a list")
        result = []
        for cite in items:
            if not isinstance(cite, dict) or cite.get("evidence_id") not in known:
                raise ValueError("Unknown evidence citation")
            quote = text(cite.get("quote"))
            evidence = known[cite["evidence_id"]]
            if not 6 <= len(quote) <= 180 or re.sub(r"\s+", "", quote) not in re.sub(r"\s+", "", evidence["text"]):
                raise ValueError(f"Quote absent/invalid at {cite['evidence_id']}: {quote[:80]}")
            result.append(evidence)
        return result

    primary_support = []
    if any(not isinstance(h, dict) or not isinstance(h.get("theme"), str) or not isinstance(h.get("mechanism"), str)
           for h in answer["hypotheses"]):
        raise ValueError("Invalid hypothesis contract")
    if answer["hypotheses"] and answer["primary_theme"] not in [h.get("theme") for h in answer["hypotheses"]]:
        raise ValueError("primary_theme must exactly match a short hypothesis theme")
    for hypothesis in answer["hypotheses"]:
        if hypothesis.get("relation_kind") not in {"MARKET_NARRATIVE", "PRODUCT_APPLICATION", "CONFIRMED_BUSINESS"}:
            raise ValueError("Unknown relation type")
        support = citations(hypothesis.get("support", []))
        citations(hypothesis.get("counter", []))
        if hypothesis["relation_kind"] == "CONFIRMED_BUSINESS" and not any(e["kind"] == "ISSUER_BODY" for e in support):
            raise ValueError("Confirmed business lacks issuer original text")
        if hypothesis.get("theme") == answer["primary_theme"]:
            primary_support.extend(support)
    business = answer.get("business_reality") or {}
    if (not isinstance(business, dict) or business.get("status") not in {"DOCUMENTED", "LIMITED", "DENIED", "UNKNOWN"}
            or not isinstance(business.get("summary"), str)):
        raise ValueError("Invalid business evidence status")
    issuer = citations(business.get("citations", []))
    if business["status"] != "UNKNOWN" and not any(e["kind"] == "ISSUER_BODY" for e in issuer):
        raise ValueError("Business claim lacks issuer original text")
    lhb = answer.get("lhb_reading") or {}
    if not isinstance(lhb, dict) or not isinstance(lhb.get("summary"), str):
        raise ValueError("Invalid trading context contract")
    if any(e["kind"] not in {"LHB", "MEDIA_BODY", "VENDOR_REASON"} for e in citations(lhb.get("citations", []))):
        raise ValueError("Trading context lacks original disclosure/media text")
    for key in ("unknowns", "next_checks"):
        if not isinstance(answer.get(key), list) or any(not isinstance(s, str) for s in answer[key]):
            raise ValueError("Missing uncertainty/check list")
    groups = {e["origin_group"] for e in primary_support if e["kind"] == "MEDIA_BODY"}
    has_issuer = any(e["kind"] == "ISSUER_BODY" for e in primary_support)
    out = copy.deepcopy(answer)
    out["processing"] = "VALIDATED"
    out["primary_has_reprint"] = any(e.get("reprint_of") for e in primary_support)
    out["verification"] = {"media_origin_groups": sorted(groups), "issuer_original": has_issuer,
                           "host_downgraded": bool((answer.get("verification") or {}).get("host_downgraded"))}
    if out["status"] == "CORROBORATED" and not (len(groups) >= 2 or (groups and has_issuer)):
        out["status"] = "INSUFFICIENT"
        out["verification"]["host_downgraded"] = True
        out["unknowns"].append("交叉证据不足：标题、摘要、旧标签、龙虎榜或转载不能凑足独立确认")
    return out


def analyze(packets, folder, cfg=None, client=None):
    cfg = cfg or model_cli.config()
    prompt = PROMPT.read_text(encoding="utf-8")
    transport = client or model_cli.call
    results, pending, meta = {}, [], {"provider": cfg["provider"], "model": cfg["model"], "calls": 0, "repairs": 0,
                                    "cache_hits": 0, "latency_s": 0.0, "errors": [], "usage": {}}
    def call(payload):
        meta["calls"] += 1
        started = time.perf_counter()
        try:
            raw, usage = transport(prompt, json.dumps(payload, ensure_ascii=False), cfg)
        finally:
            meta["latency_s"] = round(meta["latency_s"] + time.perf_counter() - started, 2)
        R._write(folder / "model_answers" / (R.digest([payload, raw])[:24] + ".json"), {"raw": raw, "usage": usage})
        for key, value in usage.items():
            if isinstance(value, (int, float)):
                meta["usage"][key] = meta["usage"].get(key, 0) + value
        return json.loads(raw)
    for packet in packets:
        stable = {k: v for k, v in packet.items() if k not in {"knowledge_cutoff", "publication_cutoff"}}
        stable["materials"] = [{k: v for k, v in e.items() if k != "collected_at"} for e in packet["materials"]]
        key = R.digest([VERSION, model_cli.VERSION, cfg["model"], prompt, stable])[:24]
        cache = folder / "analysis_cache" / (key + ".json")
        saved = R.read_json(cache)
        if saved:
            try:
                results[packet["symbol"]] = check(saved, packet)
                meta["cache_hits"] += 1
                continue
            except ValueError:
                pass
        if cfg["provider"] == "off" or not any(e["kind"] in {"MEDIA_BODY", "MEDIA_SUMMARY", "ISSUER_BODY", "VENDOR_REASON"} for e in packet["materials"]):
            results[packet["symbol"]] = unknown(packet, "模型关闭或缺少外部归因材料；不根据旧标签/龙虎榜猜题材")
        else:
            pending.append((packet, cache))
    def review_group(group, allow_split):
        try:
            response = call({"packets": [p for p, _ in group]})
            rows = response.get("rows", [])
            expected = {p["symbol"] for p, _ in group}
            if not isinstance(rows, list) or any(not isinstance(r, dict) or r.get("symbol") not in expected for r in rows):
                raise ValueError("Invalid model rows or unexpected stock")
            counts = Counter(r["symbol"] for r in rows)
            by = {r["symbol"]: r for r in rows if counts[r["symbol"]] == 1}
            for packet, cache in group:
                if packet["symbol"] not in by:
                    if allow_split and len(group) > 1:
                        meta["single_retries"] = meta.get("single_retries", 0) + 1
                        review_group([(packet, cache)], False)
                        continue
                    raise ValueError("Model omitted or duplicated requested stock")
                try:
                    try:
                        result = check(by[packet["symbol"]], packet)
                    except (ValueError, TypeError, KeyError) as exc:
                        # One bounded contract repair uses the same frozen material, never a new search or new facts.
                        meta["repairs"] += 1
                        fixed = call({"packets": [packet], "rejected_answer": by[packet["symbol"]],
                                      "repair_request": "仅修正契约或引文错误，不新增事实；不确定则INSUFFICIENT", "validation_error": str(exc)})
                        if len(fixed.get("rows", [])) != 1:
                            raise ValueError("Repair omitted/added a stock")
                        result = check(fixed["rows"][0], packet)
                    R._write(cache, result)
                except Exception as exc:
                    meta["errors"].append(packet["symbol"] + ": " + str(exc))
                    result = unknown(packet, "模型引用或格式未通过校验：" + str(exc), "VALIDATION_FAILED")
                results[packet["symbol"]] = result
        except Exception as exc:
            R._write(folder / "model_failures" / (R.digest([p for p, _ in group])[:24] + ".json"),
                     {"error": str(exc)[:300], "symbols": [p["symbol"] for p, _ in group]})
            for packet, cache in group:
                if packet["symbol"] in results:
                    continue
                if allow_split and len(group) > 1 and isinstance(exc, (ValueError, TypeError, AttributeError)):
                    meta["single_retries"] = meta.get("single_retries", 0) + 1
                    review_group([(packet, cache)], False)
                else:
                    meta["errors"].append(packet["symbol"] + ": " + str(exc)[:240])
                    results[packet["symbol"]] = unknown(packet, "模型未完成：" + str(exc)[:180], "MODEL_FAILED")
    for start in range(0, len(pending), BATCH):
        group = pending[start:start + BATCH]
        print(f"[limit_theme] model {start + 1}-{start + len(group)}/{len(pending)}", flush=True)
        review_group(group, True)
    return [results[p["symbol"]] for p in packets], meta


def run(day, root=R.OUT_ROOT, symbols=None, collector=None, cfg=None, client=None, now=None, refresh=False):
    root = Path(root)
    now = Q.stamp(now or Q.now_china())
    if day > now.date().isoformat() or (day == now.date().isoformat() and now.hour < 15):
        raise ValueError("After-close attribution requires a completed trading day")
    base = root / day.replace("-", "")
    pointer = R.read_json(base / "eod_review/latest.json")
    facts_path = Path(pointer["path"]) / "facts.json" if pointer else None
    facts = R.read_json(facts_path) if facts_path else {}
    if facts.get("as_of") != day:
        raise ValueError("No current-day frozen eod_review facts; run eod_review first")
    universe = sorted(set(facts["boards"]["U"]))
    selected = universe if symbols is None else sorted(set(symbols))
    if set(selected) - set(universe):
        raise ValueError("Requested stock is not in the frozen limit-up universe")
    folder = base / "limit_theme"
    collector = collector or Collector(folder, day, now=now, refresh=refresh)
    mapping = R.read_json(Path("concept_mapping.json")).get("mapping", {})
    market = collector.market_sources() if selected else {}
    packets = []
    for index, symbol in enumerate(selected, 1):
        info = mapping.get(symbol, {})
        stock = {"symbol": symbol, "name": info.get("name") or symbol, "height": facts["ladder"]["heights"].get(symbol, 1),
                 "legacy_concepts": info.get("concepts", []), "legacy_note": "背景标签，不是归因证据"}
        try:
            packet = collector.packet(stock, market)
        except Exception as exc:
            packet = {"as_of": day, **stock, "materials": [], "coverage": {}, "gaps": [str(exc)[:240]], "knowledge_cutoff": now.isoformat()}
        packets.append(packet)
        print(f"[limit_theme] materials {index}/{len(selected)} {symbol} count={len(packet['materials'])} gaps={len(packet['gaps'])}", flush=True)
    rows, meta = analyze(packets, folder, cfg, client)
    for row, packet in zip(rows, packets):
        row.update(name=packet["name"], height=packet["height"], legacy_concepts=packet["legacy_concepts"], coverage=packet["coverage"],
                   material_ids=[m["id"] for m in packet["materials"]], gaps=packet["gaps"], orders_enabled=False)
    counts = dict(Counter(r["status"] for r in rows))
    full = selected == universe
    status = "SUCCESS" if full and not meta["errors"] and not any(p["gaps"] for p in packets) and not counts.get("INSUFFICIENT") else "PARTIAL"
    report = {"as_of": day, "version": VERSION, "status": status, "knowledge_cutoff": now.isoformat(),
              "known_at": Q.now_china().isoformat(), "orders_enabled": False,
              "mode": "SAME_DAY_REVIEW" if day == now.date().isoformat() else "HISTORICAL_RECONSTRUCTION",
              "scope": {"universe": len(universe), "requested": len(selected), "full_limit_up_coverage": full},
              "summary": counts, "rows": rows, "materials": {p["symbol"]: p["materials"] for p in packets},
              "model": meta, "facts_source": {"path": str(facts_path), "sha256": R.file_sha(facts_path)},
              "notice": "归因与经营证据分开；交叉支持不是唯一因果或买入许可；事后采集不证明历史当时可知；模型不写概念库和账本。"}
    fingerprint = R.digest({k: v for k, v in report.items() if k not in {"knowledge_cutoff", "known_at", "model"}})
    latest = R.read_json(folder / "latest.json")
    if latest.get("fingerprint") == fingerprint and (Path(latest["path"]) / "report.json").exists():
        target = Path(latest["path"])
        report = R.read_json(target / "report.json")
    else:
        versions = [int(p.name[1:]) for p in folder.glob("v[0-9][0-9][0-9]") if p.is_dir()]
        target = folder / f"v{max(versions, default=0) + 1:03d}"
        target.mkdir(parents=True, exist_ok=False)
        R._write(target / "report.json", report)
        R._write(target / "packets.json", {"packets": packets})
        (target / "report.html").write_text(render(report), encoding="utf-8")
    # Publish the receipt only once the version is complete. Input cutoff is not availability.
    receipt_path = target / "published.json"
    receipt = R.read_json(receipt_path)
    if not receipt:
        receipt = {"as_of": day, "available_at": Q.now_china().isoformat(), "sha256": R.file_sha(target / "report.json")}
        R._write(receipt_path, receipt)
    pointer = {"as_of": day, "version": target.name, "path": str(target.resolve()), "fingerprint": fingerprint,
               **receipt, "status": status, "scope": report["scope"], "summary": counts}
    R._write(folder / "latest.json", pointer)
    R._write(folder / "status.json", {**pointer, "known_at": receipt["available_at"], "errors": meta["errors"]})
    if full:
        R._write(base / "theme_attribution.json", {**report, **receipt, "known_at": receipt["available_at"],
                                                  "report_path": str(target.resolve())})
    print(f"[limit_theme] {day} coverage={len(rows)}/{len(universe)} {counts} model_calls={meta['calls']} cached={meta['cache_hits']} -> {target / 'report.html'}", flush=True)
    return report


def render(report):
    e = lambda value: html.escape(str(value if value is not None else ""), quote=True)
    parts = []
    for row in sorted(report["rows"], key=lambda r: (-r["height"], r["symbol"])):
        sources = {m["id"]: m for m in report["materials"][row["symbol"]]}
        evidence = []
        for hypothesis in row["hypotheses"]:
            evidence.append("<h4>" + e(hypothesis.get("theme")) + "</h4><p>" + e(hypothesis.get("mechanism")) + "</p>")
            for key, label in (("support", "支持"), ("counter", "反证")):
                for cite in hypothesis.get(key, []):
                    source = sources[cite["evidence_id"]]
                    evidence.append(f'<p>{label}：{e(cite["quote"])} <a href="{e(source["url"])}" target="_blank" rel="noopener">{e(source["origin"])} / {e(source["kind"])}</a></p>')
        for source in sources.values():
            evidence.append(f'<details><summary>{e(source["kind"])} · {e(source["title"])} · {e(source.get("published_at") or source.get("event_date"))}</summary><p>采集 {e(source.get("collected_at"))} · 原始来源组 {e(source["origin_group"])}</p><pre>{e(source["text"])}</pre><a href="{e(source["url"])}" target="_blank" rel="noopener">原始来源</a></details>')
        detail = "".join(evidence) + "<h4>未知与待核</h4><p>" + e("；".join(row["unknowns"] + row["next_checks"] + row["gaps"])) + "</p>"
        parts.append(f'<tr><td><b>{e(row["name"])}</b><small>{e(row["symbol"])} · {row["height"]}板</small></td><td>{e(row["primary_theme"])}<small>{e(STATUSES[row["status"]])}</small></td><td>{e(row["business_reality"]["summary"])}</td><td>{e(row["lhb_reading"]["summary"])}</td><td>{e("；".join(row["legacy_concepts"]))}</td><td><details><summary>证据与反证 ({len(sources)})</summary>{detail}</details></td></tr>')
    return ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>涨停股概念归因</title><style>body{font:14px/1.6 "Microsoft YaHei",sans-serif;color:#202a26;background:#fff;margin:24px}h1{font-size:24px}table{border-collapse:collapse;width:100%;table-layout:fixed}th,td{border-bottom:1px solid #dce4df;padding:10px;vertical-align:top;text-align:left;overflow-wrap:anywhere}th{background:#f0f4f2}th:first-child{width:130px}small{display:block;color:#65746c}details{margin:6px 0}summary{cursor:pointer;color:#17634d}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7f6;padding:8px}a{color:#17634d}.scroll{overflow:auto}table{min-width:1000px}</style>'
            f'<header><a href="/trading">返回大富翁系统</a><h1>涨停股概念归因 · {e(report["as_of"])}</h1><p>{e(report["notice"])}</p><p>状态 {e(report["status"])} · 覆盖 {len(report["rows"])}/{report["scope"]["universe"]} · {e(report["summary"])} · 获知截止 {e(report["known_at"])} · {e(report["mode"])}</p></header>'
            '<div class="scroll"><table><thead><tr><th>股票 / 高度</th><th>主炒假说 / 交叉核验</th><th>经营关联</th><th>龙虎榜背景</th><th>旧标签（仅对照）</th><th>原文与待核</th></tr></thead><tbody>' + "".join(parts) + '</tbody></table></div></html>')

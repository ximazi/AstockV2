"""Disk-backed, cancellable bulk exports. No changes to human review records."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from threading import Lock
from uuid import uuid4
import zipfile

import pandas as pd

from . import review_workbench as RW

EXPORT_VERSION = "review_bulk_v2"
POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="review-export")
LOCK = Lock()
STATUS_LOCK = Lock()
ACTIVE = {}


def validate_ids(ids, root=RW.ROOT):
    if not isinstance(ids, list) or not ids or any(not isinstance(cid, str) for cid in ids):
        raise ValueError("至少选择一个案例")
    known = {r["id"] for r in RW.case_list(root)["cases"]}
    if len(ids) != len(set(ids)) or set(ids) - known:
        raise ValueError("案例编号重复或不在当前清单中")
    return ids


def job_dir(job_id, root):
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise ValueError("导出任务编号无效")
    return root / "exports" / job_id


def write_status(folder, state):
    # Windows will reject replacement while a polling reader holds the file open.
    with STATUS_LOCK:
        temporary = folder / "status.tmp"
        temporary.write_bytes(RW.encoded(state))
        temporary.replace(folder / "status.json")


def status(job_id, root=RW.ROOT):
    folder = job_dir(job_id, root)
    with STATUS_LOCK:
        state = json.loads((folder / "status.json").read_text(encoding="utf-8"))
    if state["status"] in {"QUEUED", "RUNNING"} and job_id not in ACTIVE:
        state.update(status="INTERRUPTED", error="服务已重启，请重新导出；已有案例缓存可复用")
    return state


def cancel(job_id, root=RW.ROOT):
    state = status(job_id, root)
    if state["status"] in {"QUEUED", "RUNNING"}:
        (job_dir(job_id, root) / "cancel").touch()
    return state


def start(ids, kind, root=RW.ROOT):
    if not isinstance(kind, str) or kind not in {"case", "review"}:
        raise ValueError("导出类型无效")
    ids = validate_ids(ids, root)
    with LOCK:
        if ACTIVE:
            raise RW.Conflict("已有批量导出在运行，请等待完成或取消")
        job_id = uuid4().hex
        folder = job_dir(job_id, root)
        folder.mkdir(parents=True)
        state = {"id": job_id, "kind": kind, "status": "QUEUED", "total": len(ids), "completed": 0,
                 "succeeded": 0, "failed": 0, "current": "", "created_at": datetime.now(timezone.utc).isoformat()}
        # Fix human revision snapshots at dispatch, not at each later case iteration.
        reviews = RW.saved_reviews(root) if kind == "review" else {}
        write_status(folder, state)
        ACTIVE[job_id] = True
        POOL.submit(run, ids, kind, reviews, state, root)
    return state.copy()


def case_archive(cid, root):
    from .review_chart import VERSION as CHART_VERSION
    from . import price_volume_context as PV, phase_events as PE, breakout_events as BE
    case = RW.detail(cid, root)
    observer_hash = hashlib.sha256(b"".join(Path(m.__file__).read_bytes() for m in (PV, PE, BE))).hexdigest()[:16]
    cache = root / "export_cache" / f"{case['case_hash']}_{EXPORT_VERSION}_{CHART_VERSION}_{observer_hash}.zip"
    cache.parent.mkdir(exist_ok=True)
    if not cache.exists():
        data = RW.archive(cid, "case", root)
        temp = cache.with_suffix(".tmp")
        temp.write_bytes(data)
        temp.replace(cache)
    return cache


def review_files(cid, summary, reviews, root):
    review = reviews.get(cid)
    reference = {"summary": summary, "case_hash": review.get("case_hash") if review else None}
    if (root / "catalog.json").exists():
        from .review_catalog import load
        catalog = load(root)
        reference.update(catalog_hash=catalog["catalog_hash"], source_row=catalog["entries"][cid]["source_row"],
                         row_hash=catalog["entries"][cid]["row_hash"])
    elif (root / "cases" / f"{cid}.json").exists():
        case = RW.detail(cid, root)
        reference.update(case_hash=case["case_hash"], source_row=case["source_row"])
    if review is None:
        review = RW.blank_review({"summary": summary, "case_hash": reference["case_hash"]})
    history = []
    if review["revision"]:
        from contextlib import closing
        import sqlite3
        with closing(sqlite3.connect(root / "reviews.sqlite3")) as conn:
            history = [json.loads(r[0]) for r in conn.execute("SELECT payload FROM reviews WHERE case_id=? AND revision<=? ORDER BY revision", (cid, review["revision"]))]
    labels = {"reviewer": "复盘人", "summary": "复盘总结", "lifecycle_notes": "生命周期判断", "entry_notes": "入场判断",
              "exit_notes": "退出判断", "ai_notes": "AI分析记录（未自动核验）", "improvements": "改进假设"}
    md = f"# {summary['name']} / {cid}\n\n保存版本 {review['revision']} / {review['status']}\n\n"
    md += "\n\n".join(f"## {label}\n\n{review[key] or '未填写'}" for key, label in labels.items())
    md += "\n\n## 人工阶段\n\n" + "\n\n".join(f"{r['start']} 至 {r['end']} / {RW.STAGES[r['stage']]}\n\n{r['note']}" for r in review["annotations"])
    files = {"review.json": RW.encoded(review), "review.md": md.encode("utf-8"),
             "review_history.json": RW.encoded(history), "case_reference.json": RW.encoded(reference)}
    selection = summary.get("exit_selection")
    if selection:
        selection = {**selection, "related_reviews": [
            {"id": r["id"], "exit_policy": r["exit_policy"], "revision": reviews[r["id"]]["revision"],
             "review_status": reviews[r["id"]]["status"]}
            for r in selection["alternatives"] if r["id"] != cid and r["id"] in reviews]}
        files["exit_selection.json"] = RW.encoded(selection)
        files["review.md"] += f"\n\n## 同信号择优口径\n\n{selection['label']}。{selection['notice']}\n".encode("utf-8")
        files["other_policy_reviews.json"] = RW.encoded([reviews[r["id"]] for r in selection["alternatives"]
                                                        if r["id"] != cid and r["id"] in reviews])
    return files, review


def run(ids, kind, reviews, state, root):
    folder = job_dir(state["id"], root)
    outcomes, combined = [], []
    try:
        listing = RW.case_list(root)
        summaries = {r["id"]: r for r in listing["cases"]}
        state["status"] = "RUNNING"
        with zipfile.ZipFile(folder / "bundle.partial", "w", zipfile.ZIP_DEFLATED, allowZip64=True) as output:
            for cid in ids:
                if (folder / "cancel").exists():
                    break
                state["current"] = cid
                write_status(folder, state)
                record = {"id": cid, "name": summaries[cid]["name"], "result": "OK"}
                try:
                    if kind == "case":
                        with zipfile.ZipFile(case_archive(cid, root)) as case_zip:
                            for name in case_zip.namelist():
                                output.writestr(f"{cid}/{name}", case_zip.read(name))
                    else:
                        files, review = review_files(cid, summaries[cid], reviews, root)
                        for name, data in files.items():
                            output.writestr(f"{cid}/{name}", data)
                        combined.append(review)
                        record.update(revision=review["revision"], review_status=review["status"])
                    state["succeeded"] += 1
                except Exception as exc:
                    record.update(result="FAILED", error=str(exc))
                    state["failed"] += 1
                outcomes.append(record)
                state["completed"] += 1
                write_status(folder, state)
            completed = {r["id"] for r in outcomes}
            outcomes.extend({"id": cid, "result": "CANCELLED"} for cid in ids if cid not in completed)
            state["status"] = "CANCELLED" if len(completed) < len(ids) else "PARTIAL" if state["failed"] else "COMPLETE"
            manifest = {**state, "version": EXPORT_VERSION, "requested_ids": ids, "results": outcomes,
                        "cancelled": len(completed) < len(ids), "notice": RW.NOTICE, "scope": listing.get("scope", "")}
            output.writestr("manifest.json", RW.encoded(manifest))
            output.writestr("summary.csv", pd.DataFrame(outcomes).to_csv(index=False).encode("utf-8-sig"))
            if kind == "review":
                output.writestr("all_reviews.json", RW.encoded(combined))
            output.writestr("README.md", "# 批量复盘导出\n\n每个案例独立目录。manifest.json列出请求清单、成功、失败和取消项，失败不冒充成功。分析使用任务提交时已保存版本；空白保持空白，不自动生成AI结论。\n".encode("utf-8"))
        (folder / "bundle.partial").replace(folder / "bundle.zip")
        state.update(status="CANCELLED" if len(completed) < len(ids) else "PARTIAL" if state["failed"] else "COMPLETE", current="",
                     download_url=f"/api/review-exports/{state['id']}/download")
    except Exception as exc:
        state.update(status="FAILED", error=str(exc))
    finally:
        write_status(folder, state)
        with LOCK:
            ACTIVE.pop(state["id"], None)

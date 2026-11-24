"""Fail-fast daily orchestrator with an auditable mainwave run status."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from . import __version__


VERSION = "lifecycle_analysis_pipeline_v2_10_20261124"
OUT_ROOT = Path("results/mainwave_engine")
REVIEW_STEPS = {"eod_review", "limit_theme", "fupan", "board_relay"}


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def _persist(status: dict, output: Path) -> None:
    _write(output / "mainwave_run_status.json", status)
    _write(OUT_ROOT / "mainwave_run_status.json", status)


def run(as_of: str, rebuild_current: bool = False,
        force_weekly_rebuild: bool = False) -> int:
    output = OUT_ROOT / as_of.replace("-", "")
    marker = output / "_SUCCESS"
    if marker.exists():
        marker.unlink()
    daily_args = ["--date", as_of] + (["--trajectory-only"] if rebuild_current else [])
    weekly_args = [sys.executable, "-m", "research.theme_ignition.weekly_reverse_trace",
                   "--end-date", as_of]
    if force_weekly_rebuild:
        weekly_args.append("--force-rebuild")
    steps = [
        ("calendar_preflight", [sys.executable, "-m", "mainwave_engine.trading.preflight", "--date", as_of]),
        ("virtual_portfolio", [sys.executable, "-m", "mainwave_engine.virtual_portfolio", "--date", as_of]),
        ("market_context", [sys.executable, "-m", "mainwave_engine.market_context", "--date", as_of]),
        ("paper_account_risk", [sys.executable, "-m", "mainwave_engine.trading.risk_daily", "--date", as_of]),
        ("daily_run", [sys.executable, "-m", "mainwave_engine.daily_run", *daily_args]),
        ("chain_pulse", [sys.executable, "-m", "mainwave_engine.chain_pulse", "--date", as_of]),
        ("p1_daily", [sys.executable, "-m", "mainwave_engine.p1_daily", "--date", as_of]),
        ("slow_carrier", [sys.executable, "-m", "mainwave_engine.slow_carrier", "--date", as_of]),
        ("mean_reversion_shadow", [sys.executable, "-m", "mainwave_engine.mean_reversion_shadow",
                                   "--date", as_of]),
        ("candidate_report", [sys.executable, "-m", "mainwave_engine.candidate_report", "--date", as_of]),
        ("theme_proposals", [sys.executable, "-m", "mainwave_engine.theme_proposals", "--date", as_of]),
        ("recent_review", [sys.executable, "-m", "mainwave_engine.recent_review", "--date", as_of]),
        ("theme_horizon", [sys.executable, "-m", "mainwave_engine.theme_horizon", "--date", as_of]),
        ("weekly_reverse_trace", weekly_args),
        ("route_scan", [sys.executable, "-m", "mainwave_engine.route_scan", "--date", as_of]),
        ("mainwave_path_shadow", [sys.executable, "-m", "research.mainwave_path", "--date", as_of]),
        ("three_direction_report", [sys.executable, "-m",
                                    "mainwave_engine.three_direction_report",
                                    "--date", as_of]),
        ("analysis_daily", [sys.executable, "-m", "mainwave_engine.analysis_daily", "--date", as_of]),
        ("trading_closure", [sys.executable, "-m", "mainwave_engine.trading.daily", "--date", as_of]),
        ("strong_trend_shadow", [sys.executable, "-m", "mainwave_engine.strong_trend", "--date", as_of]),
        ("scenario_observations", [sys.executable, "-m", "mainwave_engine.scenario_scan", "--date", as_of]),
        ("eod_review", [sys.executable, "-m", "mainwave_engine.eod_review", "--date", as_of]),
        ("limit_theme", [sys.executable, "-m", "mainwave_engine.limit_theme", "--date", as_of]),
        ("board_relay", [sys.executable, "-m", "mainwave_engine.board_relay", "--date", as_of]),
        ("fupan", [sys.executable, "-m", "mainwave_engine.fupan", "--date", as_of, "--analyze"]),
    ]
    status = {
        "as_of": as_of, "version": VERSION, "engine_version": __version__,
        "mode": "REBUILD_CURRENT" if rebuild_current else "LIVE_ADVANCE",
        "status": "RUNNING", "started_at": datetime.now().isoformat(timespec="seconds"),
        "finished_at": "", "failed_step": "", "steps": [], "warnings": [], "delivery_status": "RUNNING",
    }
    _persist(status, output)
    for index, (name, command) in enumerate(steps, 1):
        print(f"\n[mainwave_pipeline] [{index}/{len(steps)}] {name}", flush=True)
        item = {"name": name, "status": "RUNNING", "exit_code": None,
                "started_at": datetime.now().isoformat(timespec="seconds"), "finished_at": ""}
        status["steps"].append(item)
        _persist(status, output)
        if name in REVIEW_STEPS:
            _write(output / f"{name}/status.json", {"as_of": as_of, "status": "RUNNING"})
        result = subprocess.run(command, check=False)
        item["exit_code"] = result.returncode
        item["finished_at"] = datetime.now().isoformat(timespec="seconds")
        item["status"] = "SUCCESS" if result.returncode == 0 else "FAILED"
        if name == "trading_closure" and result.returncode == 0:
            for filename, label in (("jev_shadow.json", "jev_shadow"), ("closure_preflight.json", "preflight")):
                path = output / filename
                if not path.exists():
                    continue
                try:
                    health = json.loads(path.read_text(encoding="utf-8-sig"))
                except (OSError, ValueError):
                    health = {"status": "UNREADABLE"}
                item.setdefault("health", {})[label] = health.get("status")
                if health.get("as_of") != as_of or health.get("status") not in ("RECORDED", "READY"):
                    item["status"] = "WARNING"
                    item["warning"] = f"{label}: {health.get('status')}; rule-based closure retained"
                    status["warnings"].append(item["warning"])
        if name in ("strong_trend_shadow", "scenario_observations", "mainwave_path_shadow") and result.returncode != 0:
            item["status"] = "WARNING"
            item["warning"] = f"{name}: exit={result.returncode}; observation coverage incomplete"
            status["warnings"].append(item["warning"])
            _persist(status, output)
            print(f"[mainwave_pipeline] WARNING {item['warning']}", flush=True)
            continue
        if name in REVIEW_STEPS:
            try:
                review = json.loads((output / f"{name}/status.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                review = {"status": "MISSING_STATUS"}
            accepted = ("READY", "PARTIAL") if name == "board_relay" else ("SUCCESS",)
            if name == "board_relay":
                item["review_status"] = review.get("status")
                item["gaps"] = review.get("gaps") or []
            if result.returncode != 0 or review.get("status") not in accepted or review.get("as_of") != as_of:
                item["status"] = "WARNING"
                item["review_status"] = review.get("status")
                item["warning"] = f"{name}: {review.get('status')} {review.get('error') or review.get('stage') or 'review output incomplete'}"
                status["warnings"].append(item["warning"])
                _persist(status, output)
                print(f"[mainwave_pipeline] WARNING {item['warning']}; core analysis retained", flush=True)
                continue
        if result.returncode != 0:
            status["status"] = "FAILED"
            status["delivery_status"] = "FAILED"
            status["failed_step"] = name
            status["finished_at"] = datetime.now().isoformat(timespec="seconds")
            _persist(status, output)
            print(f"[mainwave_pipeline] FAILED step={name} exit={result.returncode}")
            return result.returncode or 1
        _persist(status, output)
    status["status"] = "SUCCESS"
    status["delivery_status"] = "DEGRADED" if status["warnings"] else "SUCCESS"
    status["finished_at"] = datetime.now().isoformat(timespec="seconds")
    _persist(status, output)
    marker.write_text(json.dumps({
        "as_of": as_of, "engine_version": __version__, "pipeline_version": VERSION,
        "finished_at": status["finished_at"],
        "delivery_status": status["delivery_status"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[mainwave_pipeline] core=SUCCESS delivery={status['delivery_status']} {as_of} steps={len(steps)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--rebuild-current", action="store_true")
    parser.add_argument("--force-weekly-rebuild", action="store_true",
                        help="ignore weekly limit-panel and event-replay caches")
    args = parser.parse_args()
    date = args.date if "-" in args.date else f"{args.date[:4]}-{args.date[4:6]}-{args.date[6:]}"
    return run(date, args.rebuild_current, args.force_weekly_rebuild)


if __name__ == "__main__":
    raise SystemExit(main())

"""Exercise the real PowerShell entry without network calls or production writes."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import date


ROOT = Path(__file__).resolve().parents[1]
POWERSHELL = shutil.which("powershell.exe")


@unittest.skipUnless(POWERSHELL, "Windows PowerShell is required")
class DailyFullTests(unittest.TestCase):
    def run_daily(self, switches="", fail="", paper_status="SUCCESS", stale_paper=False,
                  fixture_reports=None, as_of="2026-09-11", fundamentals_age_days=None):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shutil.copyfile(ROOT / "daily_full.ps1", root / "daily_full.ps1")
            (root / "concept_mapping.json").write_text(
                json.dumps({"generated": "2099-01-01"}), encoding="utf-8")
            compact = as_of.replace("-", "")
            reports = root / "results/mainwave_engine" / compact
            reports.mkdir(parents=True)
            fixtures = {
                "analysis_board.json": {"as_of": as_of, "summary": {"advice_counts": {"MARKET_WAIT": 1}},
                                        "records": [{"symbol": "sh_600001", "name": "Test", "route_ids": "R1",
                                                     "active_strategy": "S4", "value_tier_label": "Wait"}]},
                "mainwave_run_status.json": {"as_of": as_of, "status": "SUCCESS", "steps": []},
                "route_scan_summary.json": {"as_of": as_of, "universe": 10, "fresh": 9,
                                            "routes": {"R1": 3, "R2": 1, "R3": 0},
                                            "preparation_origins": {"PRE_ATTEMPT": 2, "": 2}},
                "virtual_positions.json": {"as_of": as_of, "records": [],
                                           "summary": {"total": 4, "status": paper_status}},
                "trading_closure_status.json": {"as_of": as_of, "status": "SUCCESS",
                    "ranking": {"total": 10, "rankable": 9, "paper_plan_eligible": 0},
                    "paper": {"status": "UNCONFIGURED"},
                    "validation": {"mature_rows": 0, "pending_rows": 27, "data_wait_rows": 0}},
            }
            fixtures.update(fixture_reports or {})
            if fundamentals_age_days is not None:
                panel = root / "results/r4_fundamentals_panel.csv"
                panel.write_text("ts_code,ann_date\n", encoding="utf-8")
                stamp = time.time() - fundamentals_age_days * 86400
                os.utime(panel, (stamp, stamp))
            for name, payload in fixtures.items():
                (reports / name).write_text(json.dumps(payload), encoding="utf-8")
            # The shim records every CLI invocation and simulates a newly written paper report.
            driver = r"""
function python {
    if ($args[0] -eq '-c') {
        & '__PYTHON__' @args
        return
    }
    $call = @($args)
    ConvertTo-Json -InputObject $call -Compress | Add-Content calls.jsonl -Encoding UTF8
    if ($call -contains 'mainwave_engine.pipeline' -and '__STALE__' -ne 'yes') {
        $file = 'results/mainwave_engine/__COMPACT__/virtual_positions.json'
        $payload = Get-Content $file -Raw
        Set-Content $file $payload -Encoding UTF8
    }
    $code = if ('__FAIL__' -and $call -contains '__FAIL__') { 7 } else { 0 }
    & $env:COMSPEC /c exit $code
}
& (Join-Path $PSScriptRoot 'daily_full.ps1') -Date __COMPACT__ __SWITCHES__
Write-Output "TEST_JEV_PROXY=$env:JEV_PROXY"
exit $LASTEXITCODE
""".replace("__FAIL__", fail).replace("__STALE__", "yes" if stale_paper else "no").replace(
                "__SWITCHES__", switches).replace("__COMPACT__", compact).replace("__PYTHON__", sys.executable.replace("'", "''"))
            (root / "driver.ps1").write_text(driver, encoding="utf-8-sig")
            process = subprocess.run([POWERSHELL, "-NoProfile", "-NonInteractive", "-File", str(root / "driver.ps1")],
                                     cwd=temp, capture_output=True, timeout=45)
            output = (process.stdout + process.stderr).decode("utf-8", errors="replace")
            calls = [json.loads(line) for line in (root / "calls.jsonl").read_text(encoding="utf-8-sig").splitlines()]
            status_path = reports / "daily_full_status.json"
            status = json.loads(status_path.read_text(encoding="utf-8-sig")) if status_path.exists() else {}
            return process.returncode, calls, status, output

    def test_default_jev_proxy_needs_no_parameter(self):
        code, _, _, output = self.run_daily("-SkipUpdate")
        self.assertEqual(code, 0, output)
        self.assertIn("TEST_JEV_PROXY=http://127.0.0.1:7890", output)

    def test_jev_proxy_parameter_overrides_default(self):
        code, _, _, output = self.run_daily("-SkipUpdate -JevProxy http://127.0.0.1:8888")
        self.assertEqual(code, 0, output)
        self.assertIn("TEST_JEV_PROXY=http://127.0.0.1:8888", output)

    def test_auction_watch_is_an_isolated_entry(self):
        code, calls, status, output = self.run_daily("-AuctionWatch")
        self.assertEqual(code, 0, output)
        self.assertEqual(calls, [["-m", "mainwave_engine.board_auction", "--date", "2026-09-11", "--watch"]])
        self.assertFalse(status)

    def test_one_command_runs_updates_pipeline_and_etfs_once(self):
        code, calls, status, output = self.run_daily()
        self.assertEqual(code, 0, output)
        self.assertEqual(status["status"], "SUCCESS", output)
        self.assertEqual([s["name"] for s in status["steps"]], [
            "stock_daily", "market_breadth", "major_indices", "sector_indices", "concept_hotspot",
            "position_monitor", "s4_opportunities", "s3_observations", "lifecycle_pipeline", "etf_tracker", "position_monitor_final",
            "intraday_reference", "daily_focus"])
        self.assertEqual(calls[-7], ["-m", "mainwave_engine.s4_scan", "--date", "2026-09-11"])
        self.assertEqual(calls[-6], ["-m", "mainwave_engine.s3_scan", "--date", "2026-09-11"])
        self.assertEqual(calls[-5], ["-m", "mainwave_engine.pipeline", "--date", "2026-09-11"])
        self.assertEqual(calls[-4], ["-m", "mainwave_engine.etf_tracker", "--date", "2026-09-11", "--refresh"])
        self.assertEqual(calls[-3], ["-m", "mainwave_engine.trading.position_monitor", "--date", "2026-09-11"])
        self.assertEqual(calls[-2], ["-m", "mainwave_engine.intraday_review", "--date", "2026-09-11", "--prepare"])
        self.assertEqual(calls[-1], ["-m", "mainwave_engine.daily_focus", "--date", "2026-09-11"])
        self.assertIn("Closure    : candidates=10", output)
        self.assertNotIn("mainwave_visual_app", str(calls))
        self.assertNotIn("strategy_tracker", str(calls))
        self.assertNotIn("--force-weekly-rebuild", str(calls))

    def test_skip_update_keeps_all_analysis_and_forwards_rebuild_flags(self):
        code, calls, status, output = self.run_daily("-SkipUpdate -RebuildMainwave -RebuildWeeklyTrace")
        self.assertEqual(code, 0, output)
        self.assertTrue(status["skip_update"])
        self.assertEqual(len(calls), 8)
        self.assertEqual(calls[3][-2:], ["--rebuild-current", "--force-weekly-rebuild"])
        self.assertNotIn("--refresh", calls[4])

    def test_auxiliary_update_failure_is_not_hidden_by_later_success(self):
        code, calls, status, output = self.run_daily(fail="update_a_share_major_indices.py")
        self.assertEqual(code, 7, output)
        self.assertEqual(status["status"], "PARTIAL")
        self.assertEqual(len(calls), 13)
        self.assertEqual(status["steps"][2]["exit_code"], 7)

    def test_fundamentals_refresh_weekly_only_for_todays_run(self):
        today = date.today().isoformat()
        code, calls, status, output = self.run_daily(as_of=today, fundamentals_age_days=10)
        self.assertEqual(code, 0, output)
        self.assertIn(["backfill_fundamentals_r4.py", "--recent", "3"], calls)
        self.assertIn("fundamentals_weekly", [s["name"] for s in status["steps"]])
        _, calls, _, _ = self.run_daily(as_of=today, fundamentals_age_days=2)
        self.assertNotIn("backfill_fundamentals_r4.py", str(calls), "7 天内更新过就跳过")
        _, calls, _, _ = self.run_daily(fundamentals_age_days=10)
        self.assertNotIn("backfill_fundamentals_r4.py", str(calls), "补跑历史日期不拉当前财报")

    def test_stock_failure_stops_downstream_and_leaves_failed_summary(self):
        code, calls, status, output = self.run_daily(fail="x_stock_data_updater.py")
        self.assertEqual(code, 7, output)
        self.assertEqual(len(calls), 1)
        self.assertEqual(status["status"], "FAILED")
        self.assertIn("downstream: NOT RUN", output)

    def test_pipeline_failure_still_runs_etfs(self):
        code, calls, status, output = self.run_daily("-SkipUpdate", fail="mainwave_engine.pipeline")
        self.assertEqual(code, 7, output)
        self.assertEqual(status["status"], "FAILED")
        self.assertEqual(calls[0][1], "mainwave_engine.trading.position_monitor", "持仓监控先于扫描独立运行")
        self.assertEqual(calls[1][1], "mainwave_engine.s4_scan", "S4 在主流程之前，复盘能读到")
        self.assertEqual(calls[-3][1], "mainwave_engine.etf_tracker")
        self.assertEqual(calls[-2][1], "mainwave_engine.strong_trend")
        self.assertEqual(calls[-1][1], "mainwave_engine.scenario_scan", "S1/S2/S5 与导出在主流程失败时仍运行")
        self.assertIn("Candidate reports may be old", output)
        self.assertNotIn("Universe   :", output)
        self.assertNotIn("mainwave_engine.daily_focus", str(calls))

    def test_failed_s4_cannot_fill_core_focus_from_leftover_report(self):
        code, calls, _, output = self.run_daily("-SkipUpdate", fail="mainwave_engine.s4_scan")
        self.assertEqual(code, 7, output)
        self.assertEqual(calls[-1], ["-m", "mainwave_engine.daily_focus", "--date", "2026-09-11", "--skip-s4"])

    def test_optional_review_warning_reaches_final_summary(self):
        code, calls, status, output = self.run_daily("-SkipUpdate", fixture_reports={
            "mainwave_run_status.json": {"as_of": "2026-09-11", "status": "SUCCESS", "steps": [],
                "delivery_status": "DEGRADED", "warnings": ["eod_review: FAILED test source unavailable"]}})
        self.assertEqual(code, 1, output)
        self.assertEqual(status["status"], "PARTIAL")
        self.assertIn("eod_review: FAILED", " ".join(status["issues"]))
        self.assertEqual(calls[-2][1], "mainwave_engine.intraday_review")
        self.assertIn("Closure    : candidates=10", output)

    def test_missing_position_data_marks_the_whole_run_partial(self):
        code, _, status, output = self.run_daily("-SkipUpdate", paper_status="PARTIAL")
        self.assertEqual(code, 1, output)
        self.assertEqual(status["status"], "PARTIAL")
        self.assertIn("Virtual positions: PARTIAL", status["issues"][0])

    def test_old_paper_report_cannot_be_presented_as_refreshed(self):
        code, _, status, output = self.run_daily("-SkipUpdate", stale_paper=True)
        self.assertEqual(code, 1, output)
        self.assertEqual(status["status"], "PARTIAL")
        self.assertIn("not refreshed", status["issues"][0])

    def test_etf_failure_marks_the_whole_run_partial(self):
        code, _, status, output = self.run_daily("-SkipUpdate", fail="mainwave_engine.etf_tracker")
        self.assertEqual(code, 7, output)
        self.assertEqual(status["status"], "PARTIAL")

    def test_summary_accepts_legacy_empty_preparation_origin_name(self):
        code, _, status, output = self.run_daily("-SkipUpdate")
        self.assertEqual(code, 0, output)
        self.assertEqual(status["status"], "SUCCESS")
        self.assertIn("Universe   : 10 | fresh=9", output)
        self.assertIn("Routes     : R1=3 R2=1 R3=0", output)
        self.assertIn("sh_600001", output)
        self.assertNotIn("Analysis summary unavailable", output)

    def test_invalid_route_summary_still_marks_the_run_partial(self):
        code, _, status, output = self.run_daily("-SkipUpdate", fixture_reports={"route_scan_summary.json": []})
        self.assertEqual(code, 1, output)
        self.assertEqual(status["status"], "PARTIAL")
        self.assertIn("Could not read route_scan_summary.json", status["issues"][0])

    def test_route_summary_for_another_date_is_not_current(self):
        code, _, status, output = self.run_daily("-SkipUpdate", fixture_reports={
            "route_scan_summary.json": {"as_of": "2026-09-10"}})
        self.assertEqual(code, 1, output)
        self.assertEqual(status["status"], "PARTIAL")
        self.assertIn("Route summary date mismatch", status["issues"][0])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

import concept_hotspot as HOTSPOT


class _FakePro:
    def limit_list_d(self, **_kwargs):
        raise AssertionError("safe_call should intercept this method")

    def ths_hot(self, **_kwargs):
        raise AssertionError("safe_call should intercept this method")

    def kpl_concept(self, **_kwargs):
        raise AssertionError("safe_call should intercept this method")

    def ths_index(self, **_kwargs):
        raise AssertionError("safe_call should intercept this method")

    def ths_member(self, **_kwargs):
        raise AssertionError("safe_call should intercept this method")


class DynamicHotMappingTests(unittest.TestCase):
    def test_active_theme_members_are_cached_for_current_day(self) -> None:
        today = datetime.now().strftime("%Y%m%d")
        pro = _FakePro()
        calls = []

        def fake_safe_call(_pro, func, **_kwargs):
            calls.append(func.__name__)
            if func.__name__ == "ths_index":
                return pd.DataFrame([{"ts_code": "885001.TI", "name": "液冷服务器"}])
            return pd.DataFrame([{"code": "000001.SZ", "name": "A"}])

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(HOTSPOT, "DYNAMIC_MAPPING_DIR", directory), \
                patch.object(HOTSPOT, "safe_call", side_effect=fake_safe_call):
            first = HOTSPOT.fetch_dynamic_hot_mapping(pro, today, ["液冷服务器"])
            second = HOTSPOT.fetch_dynamic_hot_mapping(pro, today, ["液冷服务器"])
            files = list(Path(directory).glob("*.json"))

        self.assertEqual(first["sz_000001"]["concepts"], ["液冷服务器"])
        self.assertEqual(second, first)
        self.assertEqual(calls, ["ths_index", "ths_member"])
        self.assertEqual(len(files), 1)

    def test_historical_run_does_not_create_future_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(HOTSPOT, "DYNAMIC_MAPPING_DIR", directory), \
                patch.object(HOTSPOT, "safe_call") as safe_call:
            result = HOTSPOT.fetch_dynamic_hot_mapping(
                _FakePro(), "20200102", ["液冷服务器"]
            )
            files = list(Path(directory).glob("*.json"))

        self.assertEqual(result, {})
        self.assertEqual(files, [])
        safe_call.assert_not_called()

    def test_dynamic_membership_corrects_same_day_limit_attribution(self) -> None:
        today = datetime.now().strftime("%Y%m%d")
        pro = _FakePro()

        def fake_safe_call(_pro, func, **_kwargs):
            if func.__name__ == "limit_list_d":
                return pd.DataFrame([
                    {"ts_code": "000001.SZ"}, {"ts_code": "000002.SZ"},
                ])
            if func.__name__ == "ths_hot":
                return pd.DataFrame([{"name": "液冷服务器"}])
            if func.__name__ == "kpl_concept":
                return pd.DataFrame()
            if func.__name__ == "ths_index":
                return pd.DataFrame([{"ts_code": "885001.TI", "name": "液冷服务器"}])
            if func.__name__ == "ths_member":
                return pd.DataFrame([
                    {"code": "000001.SZ", "name": "A"},
                    {"code": "000002.SZ", "name": "B"},
                ])
            raise AssertionError(func.__name__)

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(HOTSPOT, "DYNAMIC_MAPPING_DIR", directory), \
                patch.object(HOTSPOT, "HOTSPOT_FILE", str(Path(directory) / "hotspot.json")), \
                patch.object(HOTSPOT, "HOTSPOT_HISTORY_FILE",
                             str(Path(directory) / "history.json")), \
                patch.object(HOTSPOT, "load_concept_mapping", return_value={}), \
                patch.object(HOTSPOT, "safe_call", side_effect=fake_safe_call), \
                patch("builtins.print"):
            result = HOTSPOT.fetch_daily_hotspot(pro, today)

        liquid = result["concepts"]["液冷服务器"]
        self.assertEqual(liquid["limit_count"], 2)
        self.assertEqual(set(liquid["stocks"]), {"sz_000001", "sz_000002"})
        self.assertEqual(liquid["membership_source"], "dynamic_hot_mapping")


class HotspotApiFailureTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.output = Path(directory.name) / "hotspot.json"
        self.history = Path(directory.name) / "history.json"
        patches = [
            patch.object(HOTSPOT, "HOTSPOT_FILE", str(self.output)),
            patch.object(HOTSPOT, "HOTSPOT_HISTORY_FILE", str(self.history)),
            patch.object(HOTSPOT, "fetch_dynamic_hot_mapping", return_value={}),
            patch.object(HOTSPOT, "load_concept_mapping", return_value={
                "sz_000001": {"concepts": ["液冷服务器"]},
                "sz_000002": {"concepts": ["液冷服务器"]},
            }),
            patch("builtins.print"),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        sleeper = patch("time.sleep")
        self.sleep = sleeper.start()
        self.addCleanup(sleeper.stop)
        self.pro = Mock()
        self.pro.limit_list_d.return_value = pd.DataFrame([
            {"ts_code": "000001.SZ"}, {"ts_code": "000002.SZ"},
        ])
        self.pro.ths_hot.return_value = pd.DataFrame()
        self.pro.kpl_concept.return_value = pd.DataFrame([{"name": "农业"}])
        self.today = datetime.now().strftime("%Y%m%d")

    def test_permission_denial_does_not_retry(self) -> None:
        self.pro.ths_hot.side_effect = RuntimeError(
            "抱歉，您没有接口(ths_hot)访问权限"
        )
        frame = HOTSPOT.safe_call(self.pro, self.pro.ths_hot, trade_date=self.today)
        self.assertTrue(frame.empty)
        self.assertEqual(frame.attrs["api_status"], "PERMISSION_DENIED")
        self.pro.ths_hot.assert_called_once()
        self.sleep.assert_not_called()

    def test_transient_failure_still_retries(self) -> None:
        expected = pd.DataFrame([{"ts_name": "液冷服务器", "rank": 1}])
        self.pro.ths_hot.side_effect = [TimeoutError("timeout"), expected]
        frame = HOTSPOT.safe_call(self.pro, self.pro.ths_hot)
        pd.testing.assert_frame_equal(frame, expected)
        self.assertEqual(self.pro.ths_hot.call_count, 2)
        self.assertEqual([call.args[0] for call in self.sleep.call_args_list], [2, 0.3])

    def test_retry_exhaustion_is_not_an_empty_success(self) -> None:
        self.pro.ths_hot.side_effect = TimeoutError("timeout")
        frame = HOTSPOT.safe_call(self.pro, self.pro.ths_hot)
        self.assertEqual(frame.attrs["api_status"], "FAILED")
        self.assertEqual(self.pro.ths_hot.call_count, 3)

    def test_denied_ths_retains_other_sources_and_records_degradation(self) -> None:
        self.pro.ths_hot.side_effect = RuntimeError(
            "抱歉，您没有接口(ths_hot)访问权限"
        )
        result = HOTSPOT.fetch_daily_hotspot(self.pro, self.today)
        self.assertEqual(result["data_status"], "PARTIAL")
        self.assertEqual(result["sources"]["ths_hot"], {
            "status": "PERMISSION_DENIED", "rows": 0,
        })
        self.assertEqual(result["concepts"]["液冷服务器"]["limit_count"], 2)
        self.assertIn("农业", result["concepts"])
        self.assertNotIn("ths_rank", result["concepts"]["液冷服务器"])
        self.pro.ths_hot.assert_called_once()
        saved = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(saved["sources"], result["sources"])
        self.assertEqual(saved["date"], self.today)

    def test_official_ths_name_and_rank_are_used_before_top20_selection(self) -> None:
        self.pro.ths_hot.return_value = pd.DataFrame([
            {"ts_name": f"概念{rank}", "rank": str(rank)}
            for rank in range(21, 0, -1)
        ])
        result = HOTSPOT.fetch_daily_hotspot(self.pro, self.today)
        self.assertEqual(result["data_status"], "OK")
        self.assertEqual(result["concepts"]["概念1"]["ths_rank"], 1)
        self.assertEqual(result["concepts"]["概念20"]["ths_rank"], 20)
        self.assertNotIn("概念21", result["concepts"])

    def test_empty_ths_response_is_distinct_from_permission_failure(self) -> None:
        result = HOTSPOT.fetch_daily_hotspot(self.pro, self.today)
        self.assertEqual(result["sources"]["ths_hot"]["status"], "EMPTY")
        self.assertEqual(result["data_status"], "PARTIAL")

    def test_unavailable_feeds_do_not_overwrite_existing_files(self) -> None:
        self.output.write_text('{"date":"20260904"}', encoding="utf-8")
        self.history.write_text('{"history":"preserved"}', encoding="utf-8")
        for name in ("limit_list_d", "ths_hot", "kpl_concept"):
            getattr(self.pro, name).side_effect = RuntimeError("权限不足")
        with self.assertRaisesRegex(RuntimeError, "热榜数据源不可用"):
            HOTSPOT.fetch_daily_hotspot(self.pro, self.today)
        self.assertEqual(self.output.read_text(encoding="utf-8"), '{"date":"20260904"}')
        self.assertEqual(self.history.read_text(encoding="utf-8"), '{"history":"preserved"}')


if __name__ == "__main__":
    unittest.main()

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from mainwave_engine import tail_observation as O, intraday_review as R, intraday_quotes as Q
from mainwave_engine import test_intraday_decisions as fixtures


def source(day, records, status="AVAILABLE"):
    return {"as_of": day, "retrieved_at": "2026-09-18T14:20:00+08:00", "status": status, "records": records}


class ObservationTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.DecisionTests()
        fixture.setUp()
        fixture.complete_minutes()
        self.fixture = fixture
        self.report = fixture.report()
        self.sources = {
            "previous_day": "2026-09-17",
            "boards": source("2026-09-17", [{"ts_code": "600000.SH", "limit_times": 1, "name": "首板样本"},
                                           {"ts_code": "600001.SH", "limit_times": 2}]),
            "limits": source("2026-09-18", [{"ts_code": Q.ts_code(s), "up_limit": 11.} for s in fixture.symbols]),
            "leaders": source("2026-09-17", [{"leader": "sh_600000", "theme_label": "液冷", "leader_state": "LEADER_CONFIRMED"}]),
        }

    def row(self, symbol="sh_600000"):
        return next(r for r in self.report["records"] if r["symbol"] == symbol)

    def build(self):
        return O.build(self.report, self.sources)

    def test_second_board_states_are_not_orders_or_close_confirmation(self):
        for high, close, expected in ((10.5, 10.4, "SECOND_WAIT"), (11., 11., "SECOND_PRICE"), (11., 10.4, "SECOND_OPEN")):
            self.row()["quote"].update(high=high, close=close)
            observation = self.build()
            item = observation["boards"][0]
            self.assertEqual(item["state"], expected)
            self.assertEqual(item["first_board_date"], "2026-09-17")
            self.assertFalse(item["final_confirmed"])
            self.assertFalse(item["orders_enabled"])
            self.assertEqual(observation["summary"]["prior_first_boards"], 1)

    def test_new_touch_is_tomorrow_watch_not_already_confirmed_first_board(self):
        self.row("sh_600002")["quote"].update(high=11., close=11.)
        self.row("sh_600001")["quote"].update(high=11., close=11.)
        rows = {r["symbol"]: r for r in self.build()["boards"]}
        self.assertNotIn("sh_600001", rows)
        self.assertEqual(rows["sh_600002"]["group"], "NEXT_SESSION_PREP")
        self.assertEqual(rows["sh_600002"]["state"], "FIRST_TOUCH")
        self.assertIsNone(rows["sh_600002"]["first_board_date"])
        self.row("sh_600002")["quote"]["close"] = 10.8
        self.assertIn("FIRST_OPEN", [r["state"] for r in self.build()["boards"]])

    def test_incomplete_mother_pool_does_not_prove_first_board_or_denominator(self):
        self.sources["boards"]["status"] = "PARTIAL"
        self.row("sh_600002")["quote"].update(high=11., close=11.)
        obs = self.build()
        self.assertIsNone(obs["summary"]["prior_first_boards"])
        self.assertEqual(obs["summary"]["known_first_boards"], 1)
        self.assertEqual(next(r for r in obs["boards"] if r["symbol"] == "sh_600002")["state"], "TOUCH_UNCLASSIFIED")

    def test_stale_minute_only_and_inconsistent_limit_price_cannot_confirm(self):
        for change in ({"observed_at": "2026-09-18 14:30"}, {"quality": "PRICE_ONLY"}, {"high": 11.1}):
            q = self.row()["quote"].copy()
            self.row()["quote"].update(change)
            self.assertEqual(self.build()["boards"][0]["state"], "DATA_WAIT")
            self.row()["quote"] = q

    def test_wrong_day_or_future_source_is_rejected(self):
        for change in ({"as_of": "2026-09-17"}, {"retrieved_at": "2026-09-18 16:00"}):
            original = deepcopy(self.sources["limits"])
            self.sources["limits"].update(change)
            self.assertEqual(self.build()["boards"][0]["state"], "DATA_WAIT")
            self.sources["limits"] = original
        self.sources["previous_day"] = "2026-09-18"
        self.assertEqual(self.build()["boards"], [])

    def test_leadership_is_not_inferred_from_rank_or_daily_confirmed_label(self):
        obs = self.build()
        self.assertTrue(obs["leaders"])
        for row in obs["leaders"]:
            self.assertFalse(row["leadership_confirmed"])
            self.assertTrue(row["missing"])
        self.assertEqual(next(r for r in obs["leaders"] if r["symbol"] == "sh_600000")["state"], "DAILY_LEADER")

    def test_market_gate_and_existing_buy_ranking_are_unchanged(self):
        self.report["market"]["new_review_allowed"] = False
        original = deepcopy(self.report)
        obs = self.build()
        self.assertEqual(self.report, original)
        self.assertTrue(obs["boards"])
        self.assertFalse(obs["orders_enabled"])

    def test_unscanned_first_board_is_retained_as_data_wait(self):
        self.sources["boards"]["records"].append({"ts_code": "600099.SH", "limit_times": 1})
        item = next(r for r in self.build()["boards"] if r["symbol"] == "sh_600099")
        self.assertEqual(item["state"], "DATA_WAIT")
        self.assertEqual(self.build()["summary"]["prior_first_boards"], 2)

    def test_integration_publication_render_and_expiry_keep_audit_immutable(self):
        from mainwave_visual_app import app
        with tempfile.TemporaryDirectory() as tmp, patch.dict(app.config, INTRADAY_ROOT=Path(tmp)):
            f = self.fixture
            f.bundle["tail_sources"] = self.sources
            report = f.report()
            self.assertIn("tail_observation", report)
            R.check_expiry(report, f.at)
            out = R.publish(f.bundle, report, Path(tmp))
            original = (out / "review.json").read_bytes()
            self.assertIn("首板晋级观察", (out / "review.md").read_text(encoding="utf-8"))
            client = app.test_client()
            with patch.object(Q, "now_china", return_value=f.at.to_pydatetime()):
                html = client.get('/intraday?date=2026-09-18').get_data(as_text=True)
                for title in ("龙头角色证据", "今日晋级观察", "次日预备", "二板晋级待确认"):
                    self.assertIn(title, html)
            with patch.object(Q, "now_china", return_value=(f.at + pd.Timedelta(minutes=3)).to_pydatetime()):
                data = client.get('/api/intraday?date=2026-09-18').get_json()
                self.assertEqual(data['tail_observation']['mode'], 'STALE')
            self.assertEqual((out / "review.json").read_bytes(), original)

    def test_offline_older_archive_is_not_mislabeled_today_promotion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            R._write(root / 'reference/limit_list_d_20260917.json', self.sources["boards"])
            before = list(root.rglob('*'))
            result = O.offline(root, root / 'daily', '2026-09-21', '2026-09-21 12:00')
            self.assertEqual(result['mode'], 'REFERENCE_ONLY')
            self.assertEqual(result['boards'][0]['state'], 'ARCHIVED_FIRST')
            self.assertEqual(result['summary']['today_promotion'], 0)
            self.assertEqual(before, list(root.rglob('*')))


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.at = Q.stamp('2026-09-18 14:20')
        self.frame = pd.DataFrame([{'trade_date':'20260917', 'ts_code':'600000.SH', 'limit_times':1}])
        self.calls = []
        class Client:
            def query(client, api, **kwargs):
                self.calls.append((api, kwargs))
                return self.frame
        self.client = Client()

    def fetch(self):
        return O.fetch_source(self.client, 'limit_list_d', '2026-09-17', self.root, self.at)

    def test_valid_archive_fetched_once_and_keeps_full_raw_records(self):
        a, b = self.fetch(), self.fetch()
        self.assertEqual(a, b)
        self.assertEqual(a['status'], 'AVAILABLE')
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(a['records'][0]['limit_times'], 1)

    def test_rejects_wrong_date_duplicate_missing_schema_and_invalid_counts(self):
        for frame in (self.frame.assign(trade_date='20260916'), pd.concat([self.frame, self.frame]),
                      pd.DataFrame(), self.frame.assign(limit_times=1.5), self.frame.assign(limit_times=float('inf'))):
            with tempfile.TemporaryDirectory() as tmp:
                self.frame = frame
                result = O.fetch_source(self.client, 'limit_list_d', '2026-09-17', Path(tmp), self.at)
                self.assertEqual(result['status'], 'UNAVAILABLE')
                self.assertEqual(result['records'], [])

    def test_empty_schema_is_distinct_from_permission_failure(self):
        self.frame = self.frame.iloc[:0]
        self.assertEqual(self.fetch()['status'], 'EMPTY')

    def test_empty_archive_retries_after_cooldown_in_case_publication_was_delayed(self):
        populated = self.frame.copy()
        self.frame = self.frame.iloc[:0]
        self.assertEqual(self.fetch()['status'], 'EMPTY')
        self.frame = populated
        self.at += pd.Timedelta(minutes=6)
        self.assertEqual(self.fetch()['status'], 'AVAILABLE')
        self.assertEqual(len(self.calls), 2)

    def test_denial_cached_briefly_without_fabricated_zero_pool(self):
        with patch.object(self.client, 'query', side_effect=Q.ApiError('limit_list_d', 40203, 'denied')) as query:
            self.assertEqual(self.fetch()['status'], 'UNAVAILABLE')
            self.fetch()
            self.assertEqual(query.call_count, 1)

    def test_current_unfinished_daily_list_is_never_requested(self):
        with self.assertRaises(ValueError):
            O.fetch_source(self.client, 'limit_list_d', '2026-09-18', self.root, self.at)
        self.assertEqual(self.calls, [])

    def test_partial_response_never_claims_complete_mother_pool(self):
        self.frame = pd.DataFrame([{'trade_date':'20260917', 'ts_code':f'{600000+i}.SH', 'limit_times':1} for i in range(2500)])
        self.assertEqual(self.fetch()['status'], 'PARTIAL')


if __name__ == '__main__':
    unittest.main()

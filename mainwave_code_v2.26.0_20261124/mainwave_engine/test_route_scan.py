import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mainwave_engine import route_scan as RS
from mainwave_engine.route_scan import route_ids, assess_routes


class RouteScanTests(unittest.TestCase):
    def row(self):
        return {"symbol": "sh_600001", "data_status": "FRESH", "eligible": True,
                "prepared": False, "prepared_before_attempt": False,
                "breakout_status": "ACCEPTED", "acceptance_age": 0}

    def chain(self):
        return {"as_of": "2026-09-09", "theme_id": "tight_chain", "leader": "sh_600002",
                "anchor_date": "2026-09-07", "theme_lifecycle": "FERMENTING",
                "anchor_age_sessions": 2, "lag_gap_atr": 1.2}

    def test_known_ignition_stock_does_not_need_theme_trigger(self):
        self.assertEqual(route_ids(self.row(), [], "2026-09-09")[0], ["R1"])

    def test_preparation_watch_does_not_require_hot_theme(self):
        row = {**self.row(), "prepared": True, "breakout_status": "NONE", "acceptance_age": None}
        self.assertEqual(route_ids(row, [], "2026-09-09")[0], ["R2"])

    def test_propagation_requires_dated_distinct_leader_and_lag(self):
        chain = self.chain()
        self.assertIn("R3", route_ids(self.row(), [chain], "2026-09-09")[0])
        for change in ({"as_of": "2026-09-10"}, {"leader": "sh_600001"},
                       {"theme_lifecycle": "RETREAT"}, {"anchor_age_sessions": 0},
                       {"anchor_date": "2026-09-10"}, {"anchor_date": ""}, {"lag_gap_atr": ""}):
            self.assertNotIn("R3", route_ids(self.row(), [{**chain, **change}], "2026-09-09")[0])

    def test_stale_data_cannot_enter_any_route(self):
        self.assertEqual(route_ids({**self.row(), "data_status": "STALE"}, [self.chain()], "2026-09-09")[0], [])

    def test_propagation_preparation_gate_does_not_hide_supply_risk(self):
        row = {**self.row(), "accepted_bias2": True, "supply_warning": True}
        result = assess_routes(row, "accepted_bias2", ["R3"])
        self.assertFalse(result["confirmation_pass"])
        self.assertFalse(result["route_requirements_met"])
        self.assertEqual(result["status"], "RISK_WATCH")

    def test_propagation_cannot_borrow_price_confirmation_without_preparation(self):
        row = {**self.row(), "accepted_bias2": True}
        self.assertEqual(assess_routes(row, "accepted_bias2", ["R3"])["status"], "WAIT_ACCEPTANCE")
        self.assertTrue(assess_routes(row, "accepted_bias2", ["R1"])["confirmation_pass"])

    def test_failed_old_episode_cannot_supply_current_propagation_preparation(self):
        row = {**self.row(), "prepared_before_attempt": True, "breakout_status": "FAILED"}
        self.assertFalse(assess_routes(row, "accepted_bias2", ["R3"])["route_requirements_met"])

    def test_incomplete_propagation_does_not_veto_independent_r1(self):
        row = {**self.row(), "accepted_bias2": True, "date": "2026-09-09", "breakout_id": "one",
               "breakout_accepted_at": "2026-09-09", "demand": True, "bias_atr": 1.}
        routes, _ = route_ids(row, [self.chain()], "2026-09-09")
        self.assertEqual(routes, ["R1", "R3"])
        self.assertTrue(assess_routes(row, "accepted_bias2", routes)["confirmation_pass"])

    def test_summary_counts_missing_origins_without_empty_json_keys_or_lost_cases(self):
        rows = [{**self.row(), "symbol": f"sh_60000{i}", "cache_hit": True,
                 "bias_atr": 1., "ema20_bias": 1., "ret5": .01, "ret20": .02,
                 "preparation_origin": origin, "preparation_first_seen": "", "probe_at": "",
                 "breakout_attempt_at": "2026-09-08", "breakout_accepted_at": "2026-09-09"}
                for i, origin in enumerate(("", None, "PRE_ATTEMPT", "UNKNOWN"), 1)]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "data").mkdir()
            for row in rows:
                (root / "data" / f"{row['symbol']}_daily.csv").write_text("date,close\n", encoding="utf-8")

            def rooted_path(value):
                path = Path(value)
                return path if path.is_absolute() else root / path

            with patch.object(RS, "Path", side_effect=rooted_path), patch.object(RS, "ProcessPoolExecutor") as pool, \
                    patch.object(RS.TS, "load_mapping_as_of", return_value=({}, {})), \
                    patch.object(RS, "assess_routes", return_value={"status": "TREND_WATCH", "risks": [],
                        "route_requirements_met": True, "independent_price_evidence": False, "confirmation_pass": False}):
                pool.return_value.__enter__.return_value.map.return_value = rows
                RS.run("2026-09-09")
            result = json.loads((root / "results/mainwave_engine/20260909/route_scan_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(result["preparation_origins"], {"UNKNOWN": 3, "PRE_ATTEMPT": 1})
        self.assertEqual(sum(result["preparation_origins"].values()), result["candidates"])
        self.assertEqual(result["routes"]["R1"], 4)


if __name__ == "__main__":
    unittest.main()

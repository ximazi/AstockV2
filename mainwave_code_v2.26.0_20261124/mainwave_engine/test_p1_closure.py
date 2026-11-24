from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import pandas as pd

from mainwave_engine import capital_episode as CE
from mainwave_engine import carrier_quality as CQ
from mainwave_engine import candidate_episode as CAND
from mainwave_engine import p1_daily as P1


CALENDAR = [f"2026-08-{day:02d}" for day in range(1, 32)]


def episode(symbol: str, start: str, stage: str, retention: str) -> dict:
    return {
        "episode_id": f"{symbol}:{start}:1", "symbol": symbol, "start_date": start,
        "shadow_stage": stage, "retention_state": retention,
        "production_state": "FORMING", "valid_until": "T+10 sessions",
        "evidence": {
            "event": {"first_seen": start, "last_seen": start},
            "supply": {"first_seen": start, "last_seen": start},
        },
    }


class CapitalEpisodeTests(unittest.TestCase):
    def test_async_evidence_accumulates_without_weighted_score(self) -> None:
        first = {"symbol": "sh_600000", "state": "FORMING", "data_fresh": True,
                 "health": True, "imp": True, "imp_age": 0,
                 "supply_ready": False, "structure_ready": False}
        active, ended = CE.update_episode(None, first, "2026-08-01", CALENDAR)
        self.assertIsNone(ended)
        self.assertEqual(active["shadow_stage"], "PROBED")
        self.assertEqual(active["retention_state"], "UNCERTAIN")

        supply = {"symbol": "sh_600000", "state": "FORMING", "data_fresh": True,
                  "health": True, "imp": False, "supply_ready": True,
                  "structure_ready": False}
        active, _ = CE.update_episode(active, supply, "2026-08-05", CALENDAR)
        self.assertEqual(active["shadow_stage"], "ABSORBING")
        self.assertEqual(active["retention_state"], "RETAINED")

        structure = {"symbol": "sh_600000", "state": "FORMING", "data_fresh": True,
                     "health": True, "imp": False, "supply_ready": False,
                     "structure_ready": True}
        active, _ = CE.update_episode(active, structure, "2026-08-08", CALENDAR)
        self.assertEqual(active["shadow_stage"], "COILED")

    def test_explicit_failure_closes_episode(self) -> None:
        current = episode("sh_600000", "2026-08-01", "ABSORBING", "RETAINED")
        row = {"symbol": "sh_600000", "state": "FAILED", "data_fresh": True,
               "health": False, "supply_ready": False, "structure_ready": False}
        active, ended = CE.update_episode(current, row, "2026-08-02", CALENDAR)
        self.assertIsNone(active)
        self.assertEqual(ended["failure_reason"], "STRUCTURE_FAILED")
        self.assertEqual(ended["retention_state"], "EXITED")

    def test_same_day_closed_episode_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store_path = Path(directory) / "episodes.json"
            row = {"symbol": "sh_600000", "state": "FAILED", "data_fresh": True,
                   "health": False, "imp": True, "imp_age": 0,
                   "supply_ready": False, "structure_ready": False}
            first, first_closed = CE.update_store([row], "2026-08-02", CALENDAR, store_path)
            second, second_closed = CE.update_store([row], "2026-08-02", CALENDAR, store_path)
            self.assertEqual(len(first["closed"]), 1)
            self.assertEqual(len(second["closed"]), 1)
            self.assertEqual(first_closed[0]["episode_id"], second_closed[0]["episode_id"])


class IntersectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.theme = {
            "id": "demo", "label": "Demo", "active_from": "2026-08-01",
            "members": {"sh_600001": "A", "sh_600002": "B", "sh_600003": "C"},
        }
        self.metrics = {
            "sh_600001": {"data_fresh": True, "r20": 10.0, "r5": 2.0,
                           "amount_ratio": 1.0, "close": 10.0, "drawdown20": -1.0,
                           "repair5": 4.0, "execution_state": "EXECUTABLE",
                           "extension_state": "NOT_EXTENDED", "distribution_risk": False,
                           "trigger_distance_pct": 2.0, "stop_distance_pct": 5.0},
            "sh_600002": {"data_fresh": True, "r20": 5.0, "r5": 1.0,
                           "amount_ratio": 1.0, "close": 10.0, "drawdown20": -2.0,
                           "repair5": 3.0, "execution_state": "EXECUTABLE",
                           "extension_state": "NOT_EXTENDED", "distribution_risk": False,
                           "trigger_distance_pct": 3.0, "stop_distance_pct": 6.0},
            "sh_600003": {"data_fresh": True, "r20": 3.0, "r5": 1.0,
                           "amount_ratio": 1.0, "close": 10.0, "drawdown20": -3.0,
                           "repair5": 2.0, "execution_state": "EXECUTABLE",
                           "extension_state": "NOT_EXTENDED", "distribution_risk": False,
                           "trigger_distance_pct": 4.0, "stop_distance_pct": 7.0},
        }
        self.attention = {"attention_level": "MEDIUM", "attention_slope": "RISING",
                          "crowding_state": "NOT_CROWDED"}

    def test_uncertain_seed_stays_out_of_b_focus(self) -> None:
        episodes = {"sh_600001": episode("sh_600001", "2026-08-01", "ABSORBING", "UNCERTAIN")}
        snapshot = {"stage": "SEED", "cycle": 1, "stage_since": "2026-08-01",
                    "last_ignition": ""}
        rows, propagation, _ = P1._theme_rows(
            self.theme, snapshot, self.attention, episodes, {}, "2026-08-10", "",
            self.metrics)
        self.assertEqual(rows, [])
        self.assertEqual(propagation["leader_state"], "LEADER_SEED")

    def test_retained_seed_enters_b_focus(self) -> None:
        episodes = {"sh_600001": episode("sh_600001", "2026-08-01", "ABSORBING", "RETAINED")}
        snapshot = {"stage": "SEED", "cycle": 1, "stage_since": "2026-08-01",
                    "last_ignition": ""}
        rows, _, _ = P1._theme_rows(
            self.theme, snapshot, self.attention, episodes, {}, "2026-08-10", "",
            self.metrics)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["intersection_class"], "B")

    def test_ex_leader_responses_confirm_a_without_self_count(self) -> None:
        episodes = {
            "sh_600001": episode("sh_600001", "2026-08-01", "COILED", "RETAINED"),
            "sh_600002": episode("sh_600002", "2026-08-03", "ABSORBING", "RETAINED"),
            "sh_600003": episode("sh_600003", "2026-08-04", "ABSORBING", "RETAINED"),
        }
        snapshot = {"stage": "FERMENTING", "cycle": 1, "stage_since": "2026-08-03",
                    "last_ignition": ""}
        rows, propagation, responses = P1._theme_rows(
            self.theme, snapshot, self.attention, episodes, {}, "2026-08-10", "",
            self.metrics)
        self.assertEqual(propagation["theme_signal_ex_leader"], 2)
        self.assertEqual(propagation["leader_state"], "LEADER_CONFIRMED")
        self.assertEqual(len(responses), 2)
        self.assertTrue(all(row["follower"] != row["leader"] for row in responses))
        self.assertEqual(rows[0]["intersection_class"], "A")

    def test_extended_leader_does_not_block_qualified_low_position_followers(self) -> None:
        episodes = {
            "sh_600001": episode("sh_600001", "2026-08-01", "COILED", "RETAINED"),
            "sh_600002": episode("sh_600002", "2026-08-03", "ABSORBING", "RETAINED"),
            "sh_600003": episode("sh_600003", "2026-08-04", "ABSORBING", "RETAINED"),
        }
        metrics = {symbol: dict(values) for symbol, values in self.metrics.items()}
        metrics["sh_600001"]["extension_state"] = "EXTENDED"
        metrics["sh_600001"]["r20"] = 40.0
        snapshot = {"stage": "FERMENTING", "cycle": 1, "stage_since": "2026-08-03",
                    "last_ignition": ""}
        rows, propagation, _ = P1._theme_rows(
            self.theme, snapshot, self.attention, episodes, {}, "2026-08-10", "",
            metrics)
        self.assertEqual(propagation["leader"], "sh_600001")
        self.assertEqual({row["symbol"] for row in rows}, {"sh_600002", "sh_600003"})
        self.assertTrue(all(row["carrier_qualification"] == "QUALIFIED_LURK" for row in rows))

    def test_retreat_blocks_current_episode_not_leader_observation(self) -> None:
        episodes = {"sh_600001": episode("sh_600001", "2026-08-01", "ABSORBING", "RETAINED")}
        snapshot = {"stage": "RETREAT", "cycle": 1, "stage_since": "2026-08-09",
                    "last_ignition": "2026-07-01"}
        rows, propagation, _ = P1._theme_rows(
            self.theme, snapshot, self.attention, episodes, {}, "2026-08-10", "",
            self.metrics)
        self.assertEqual(rows, [])
        self.assertEqual(propagation["leader"], "sh_600001")

    def test_theme_episode_id_stays_stable_within_cycle(self) -> None:
        episodes = {"sh_600001": episode("sh_600001", "2026-08-01", "ABSORBING", "RETAINED")}
        seed = {"stage": "SEED", "cycle": 2, "stage_since": "2026-08-01",
                "last_ignition": "2026-06-01"}
        _, first, _ = P1._theme_rows(
            self.theme, seed, self.attention, episodes, {}, "2026-08-10", "",
            self.metrics)
        fermenting = {"stage": "FERMENTING", "cycle": 2, "stage_since": "2026-08-11",
                      "last_ignition": "2026-06-01"}
        _, second, _ = P1._theme_rows(
            self.theme, fermenting, self.attention, episodes, {}, "2026-08-12",
            first["leader"], self.metrics, first["theme_episode_id"], first["cycle"])
        self.assertEqual(first["theme_episode_id"], second["theme_episode_id"])

    def test_candidate_episode_tracks_b_to_a_and_theme_terminal(self) -> None:
        base = {
            "theme_id": "demo", "theme_label": "Demo", "theme_episode_id": "demo:C1:2026-08-01",
            "symbol": "sh_600001", "name": "A", "episode_id": "stock-episode-1",
            "role": "information_leader", "production_state": "FORMING",
            "retention_state": "RETAINED", "valid_for_sessions": 10,
            "rank_global": 1, "rank_in_theme": 1,
        }
        themes = pd.DataFrame([{
            "theme_id": "demo", "theme_episode_id": "demo:C1:2026-08-01",
            "leader": "sh_600001", "crowding_state": "NOT_CROWDED", "theme_lifecycle": "SEED",
        }])
        capital = {"active": {"sh_600001": {
            "episode_id": "stock-episode-1", "retention_state": "RETAINED",
        }}, "closed": []}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store_path = root / "store.json"
            events_path = root / "events.csv"
            outcomes_path = Path(directory) / "outcomes.csv"
            first = pd.DataFrame([{**base, "intersection_class": "B"}])
            CAND.update(first, first, themes, capital, "2026-08-10", root,
                        store_path, events_path, outcomes_path)
            second = pd.DataFrame([{**base, "intersection_class": "A"}])
            _, outcomes, funnel = CAND.update(
                second, second, themes, capital, "2026-08-11", root,
                store_path, events_path, outcomes_path)
            self.assertEqual(funnel["b_to_a"], 1)
            self.assertEqual(outcomes.iloc[0]["current_state"], "A_CONFIRMED")

            next_theme = themes.copy()
            next_theme["theme_episode_id"] = "demo:C2:2026-08-12"
            _, outcomes, _ = CAND.update(
                pd.DataFrame(), pd.DataFrame(), next_theme, capital, "2026-08-12", root,
                store_path, events_path, outcomes_path)
        self.assertEqual(outcomes.iloc[0]["outcome"], "THEME_FAILED")


class CarrierQualityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.metrics = {
            "data_fresh": True, "close": 10.0, "r5": 3.0, "r20": 8.0,
            "ma20_bias": 4.0, "drawdown20": -2.0, "repair5": 3.0,
            "execution_state": "EXECUTABLE", "extension_state": "NOT_EXTENDED",
            "distribution_risk": False, "trigger_distance_pct": 2.0,
            "stop_distance_pct": 5.0,
        }

    def test_all_hard_gates_are_required(self) -> None:
        uncertain = episode("sh_600001", "2026-08-01", "COILED", "UNCERTAIN")
        result = CQ.evaluate(uncertain, self.metrics, "HOT", "FORMING")
        self.assertEqual(result["carrier_qualification"], "RETENTION_UNPROVEN")
        self.assertFalse(result["hard_gate_pass"])

    def test_retained_coiled_executable_is_qualified(self) -> None:
        retained = episode("sh_600001", "2026-08-01", "COILED", "RETAINED")
        result = CQ.evaluate(retained, self.metrics, "HOT", "FORMING")
        self.assertEqual(result["carrier_qualification"], "QUALIFIED_LURK")
        self.assertEqual(result["carrier_tier"], "CORE_LURK")

    def test_extension_cannot_be_averaged_away(self) -> None:
        retained = episode("sh_600001", "2026-08-01", "READY", "RETAINED")
        metrics = {**self.metrics, "r20": 45.0, "extension_state": "EXTENDED"}
        result = CQ.evaluate(retained, metrics, "HYBRID", "READY")
        self.assertEqual(result["carrier_qualification"], "EXTENDED_TOO_LATE")
        self.assertFalse(result["hard_gate_pass"])


if __name__ == "__main__":
    unittest.main()

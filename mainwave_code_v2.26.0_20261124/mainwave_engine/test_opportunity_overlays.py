from __future__ import annotations

import unittest

import pandas as pd

from mainwave_engine import mean_reversion_shadow as MR
from mainwave_engine import slow_carrier as SC


class SlowCarrierLifecycleTests(unittest.TestCase):
    def test_valid_lag_advances_only_after_response(self) -> None:
        common = {
            "prepared": True,
            "data_fresh": True,
            "theme_lifecycle": "FERMENTING",
            "crowded": False,
            "anchor_age": 2,
            "as_of": "2026-09-03",
            "hard_gate_pass": True,
            "failed": False,
        }
        self.assertEqual(SC.classify_stage(
            **common, current_valid_lag=True, first_response_date=""
        ), "VALID_LAG")
        self.assertEqual(SC.classify_stage(
            **common, current_valid_lag=True, first_response_date="2026-09-03"
        ), "FIRST_RESPONSE")
        self.assertEqual(SC.classify_stage(
            **common, current_valid_lag=False, first_response_date="2026-09-02"
        ), "READY")

    def test_low_price_cannot_replace_capital_and_supply_evidence(self) -> None:
        stage = SC.classify_stage(
            prepared=False, data_fresh=True, theme_lifecycle="FERMENTING",
            crowded=False, anchor_age=1, current_valid_lag=True,
            first_response_date="2026-09-03", as_of="2026-09-03",
            hard_gate_pass=False, failed=False,
        )
        self.assertEqual(stage, "NOT_PREPARED")

    def test_window_and_candidate_caps_are_hard(self) -> None:
        expired = SC.classify_stage(
            prepared=True, data_fresh=True, theme_lifecycle="FERMENTING",
            crowded=False, anchor_age=9, current_valid_lag=True,
            first_response_date="", as_of="2026-09-03",
            hard_gate_pass=True, failed=False,
        )
        self.assertEqual(expired, "EXPIRED")
        rows = []
        for theme in range(6):
            for item in range(4):
                rows.append({
                    "theme_id": f"t{theme}", "symbol": f"s{theme}_{item}",
                    "slow_stage": "VALID_LAG", "carrier_tier": "CORE_LURK",
                    "response_evidence_count": 2, "lag_gap_atr": item + 1,
                })
        focus = SC._cap_focus(pd.DataFrame(rows))
        self.assertEqual(len(focus), SC.MAX_GLOBAL)
        self.assertLessEqual(int(focus.groupby("theme_id").size().max()), SC.MAX_PER_THEME)


class MeanReversionShadowTests(unittest.TestCase):
    def test_frozen_config_cannot_leak_into_an_earlier_dated_run(self) -> None:
        action = MR.classify_action(
            enabled=True, config_effective=False, data_fresh=True,
            market_context="BEAR", market_temperature="NORMAL",
            active_contexts={"BEAR"}, oversold=True, near_trigger=True,
            cross_age=0, mode="NEXT_OPEN", ttl=0, confirm_date="",
            as_of="2026-09-02",
        )
        self.assertEqual(action, "CONFIG_NOT_EFFECTIVE_YET")

    def test_unvalidated_storage_threshold_cannot_emit_a_signal(self) -> None:
        action = MR.classify_action(
            enabled=False, data_fresh=True, market_context="BEAR",
            market_temperature="NORMAL", active_contexts={"BEAR"},
            oversold=True, near_trigger=True, cross_age=0, mode="NEXT_OPEN",
            ttl=0, confirm_date="", as_of="2026-09-03",
        )
        self.assertEqual(action, "NO_VALIDATED_THRESHOLD")

    def test_panic_and_stale_crossings_never_promote(self) -> None:
        panic = MR.classify_action(
            enabled=True, data_fresh=True, market_context="PANIC",
            market_temperature="PANIC", active_contexts={"PANIC"},
            oversold=True, near_trigger=True, cross_age=0, mode="NEXT_OPEN",
            ttl=0, confirm_date="", as_of="2026-09-03",
        )
        stale = MR.classify_action(
            enabled=True, data_fresh=True, market_context="BEAR",
            market_temperature="NORMAL", active_contexts={"BEAR"},
            oversold=True, near_trigger=True, cross_age=3, mode="NEXT_OPEN",
            ttl=0, confirm_date="", as_of="2026-09-03",
        )
        self.assertEqual(panic, "SHADOW_WATCH_PANIC_NO_ENTRY")
        self.assertEqual(stale, "OVERSOLD_STALE")


if __name__ == "__main__":
    unittest.main()

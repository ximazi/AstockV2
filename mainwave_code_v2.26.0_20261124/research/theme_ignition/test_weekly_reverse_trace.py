from __future__ import annotations

import tempfile
import unittest
import json
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from mainwave_engine import carrier_quality as CQ
from mainwave_engine.daily_run import write_limit_up_snapshot
from research.theme_ignition.weekly_reverse_trace import (
    REPLAY_COMPATIBLE_CARRIER_VERSIONS,
    REPLAY_CACHE_VERSION,
    VERSION,
    _cached_event_frames,
    _coverage,
    _load_event_cache,
    _load_replay_source,
    _panel_from_snapshot,
    _write_event_cache,
    detect_events,
    load_or_build_limit_panel,
)


class WeeklyReverseTraceTests(unittest.TestCase):
    def test_coverage_does_not_treat_cached_nan_as_observed_evidence(self) -> None:
        events = pd.DataFrame([{
            "date": "2026-08-31", "theme": "raw", "event_type": "REIGNITION",
        }])
        stocks = pd.DataFrame([{
            "raw_theme": "raw", "t0": "2026-08-31", "role": "IGNITION_CARRIER",
            "first_probe": float("nan"), "first_retained": float("nan"),
            "first_qualified": float("nan"), "pre_qualification": float("nan"),
        }])

        row = _coverage(events, stocks).iloc[0]

        self.assertEqual(row["carriers_pre_probed"], 0)
        self.assertEqual(row["carriers_pre_retained"], 0)
        self.assertEqual(row["carriers_pre_qualified"], 0)

    def test_daily_limit_snapshot_uses_existing_frames(self) -> None:
        frames = {
            "sh_600001": pd.DataFrame([
                {"date": "2026-09-01", "pct_chg": 9.9},
            ]),
            "sh_688001": pd.DataFrame([
                {"date": "2026-09-01", "pct_chg": 19.8},
            ]),
            "sz_000001": pd.DataFrame([
                {"date": "2026-09-01", "pct_chg": 5.0},
            ]),
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "limit_up_snapshot.csv"

            count = write_limit_up_snapshot(frames, "2026-09-01", path)
            snapshot = pd.read_csv(path, encoding="utf-8-sig")

        self.assertEqual(count, 2)
        self.assertEqual(set(snapshot["symbol"]), {"sh_600001", "sh_688001"})

    def test_incremental_panel_reuses_prior_day_and_only_reads_new_snapshot(self) -> None:
        calendar = [date.strftime("%Y-%m-%d") for date in pd.bdate_range(
            end="2026-09-01", periods=40
        )]
        end_date = calendar[-1]
        week_start = end_date
        required_dates = calendar[-31:]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prior = root / calendar[-2].replace("-", "") / "weekly_reverse_trace"
            prior.mkdir(parents=True)
            pd.DataFrame([{
                "date": calendar[-2], "theme": "theme_x", "member_count": 2,
                "ignition_threshold": 3, "limit_count": 1, "limit_stocks": "a",
            }]).to_csv(prior / "limit_panel.csv", index=False, encoding="utf-8-sig")
            (prior / "limit_panel_manifest.json").write_text(
                "{\n"
                '  "version": "weekly_limit_panel_v1_20260901",\n'
                '  "mapping_sha256": "mapping-hash",\n'
                f'  "processed_dates": {pd.Series(required_dates[:-1]).to_json(orient="values")}\n'
                "}",
                encoding="utf-8",
            )
            today = root / end_date.replace("-", "")
            today.mkdir()
            pd.DataFrame([{
                "date": end_date, "symbol": "b", "pct_chg": 10.0, "limit_pct": 10.0,
            }]).to_csv(today / "limit_up_snapshot.csv", index=False, encoding="utf-8-sig")
            output = today / "weekly_reverse_trace"
            mapping = {"a": {"concepts": ["theme_x"]}, "b": {"concepts": ["theme_x"]}}

            with patch("research.theme_ignition.weekly_reverse_trace.D.list_universe",
                       return_value=["a", "b"]), patch(
                "research.theme_ignition.weekly_reverse_trace._mapping_hash",
                return_value="new-mapping-hash",
            ), patch(
                "research.theme_ignition.weekly_reverse_trace._panel_rows_for_dates",
                side_effect=AssertionError("full market scan should not run"),
            ):
                panel, _, meta = load_or_build_limit_panel(
                    calendar, week_start, end_date, mapping,
                    {"path": "mapping.json", "generated": "2026-09-01"}, output,
                    output_root=root,
                )

            self.assertEqual(meta["mode"], "INCREMENTAL")
            self.assertEqual(meta["snapshot_dates"], [end_date])
            self.assertEqual(meta["scanned_dates"], [])
            self.assertEqual(meta["source_mapping_sha256"], "mapping-hash")
            self.assertEqual(meta["mapping_sha256"], "new-mapping-hash")
            self.assertEqual(set(panel["date"]), {calendar[-2], end_date})

    def test_limit_panel_can_be_derived_from_daily_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "20260901"
            output.mkdir()
            pd.DataFrame([
                {"date": "2026-09-01", "symbol": "a", "pct_chg": 10.0, "limit_pct": 10.0},
                {"date": "2026-09-01", "symbol": "b", "pct_chg": 20.0, "limit_pct": 20.0},
            ]).to_csv(output / "limit_up_snapshot.csv", index=False, encoding="utf-8-sig")
            mapping = {
                "a": {"concepts": ["theme_x", "theme_y"]},
                "b": {"concepts": ["theme_x"]},
            }

            panel = _panel_from_snapshot(
                "2026-09-01", mapping, Counter({"theme_x": 50, "theme_y": 10}), root
            )

            self.assertIsNotNone(panel)
            keyed = {row.theme: row for row in panel.itertuples(index=False)}
            self.assertEqual(keyed["theme_x"].limit_count, 2)
            self.assertEqual(keyed["theme_x"].limit_stocks, "a|b")
            self.assertEqual(keyed["theme_x"].ignition_threshold, 3)
            self.assertEqual(keyed["theme_y"].limit_count, 1)

    def test_event_cache_requires_exact_members_and_refreshes_current_quality(self) -> None:
        source = {
            "stocks": pd.DataFrame([
                {
                    "raw_theme": "raw", "theme": "resolved", "t0": "2026-08-31",
                    "symbol": "a", "current_qualification": "OLD",
                    "current_carrier_tier": "OLD", "current_gate_failures": "OLD",
                },
                {
                    "raw_theme": "raw", "theme": "resolved", "t0": "2026-08-31",
                    "symbol": "b", "current_qualification": "OLD",
                    "current_carrier_tier": "OLD", "current_gate_failures": "OLD",
                },
            ]),
            "traces": pd.DataFrame([
                {"raw_theme": "raw", "t0": "2026-08-31", "symbol": "a", "date": "2026-08-01"},
                {"raw_theme": "raw", "t0": "2026-08-31", "symbol": "b", "date": "2026-08-01"},
            ]),
        }
        event = {"raw_theme": "raw", "theme": "resolved", "date": "2026-08-31"}
        quality = {
            "a": {
                "carrier_qualification": "QUALIFIED", "carrier_tier": "CORE",
                "gate_failures": "",
            }
        }

        cached = _cached_event_frames(source, event, ["a", "b"], quality)

        self.assertIsNotNone(cached)
        summary, trace = cached
        a = summary[summary["symbol"] == "a"].iloc[0]
        b = summary[summary["symbol"] == "b"].iloc[0]
        self.assertEqual(a["current_qualification"], "QUALIFIED")
        self.assertEqual(a["current_carrier_tier"], "CORE")
        self.assertEqual(b["current_qualification"], "NO_ACTIVE_EPISODE")
        self.assertEqual(len(trace), 2)
        self.assertIsNone(_cached_event_frames(source, event, ["a"], quality))

    def test_persistent_event_cache_is_exact_and_survives_empty_trace(self) -> None:
        event = {
            "raw_theme": "raw", "theme": "resolved", "date": "2026-08-31",
            "limit_stocks": "a",
        }
        summary = pd.DataFrame([{
            "raw_theme": "raw", "theme": "resolved", "t0": "2026-08-31",
            "symbol": "a", "current_qualification": "OLD",
            "current_carrier_tier": "OLD", "current_gate_failures": "OLD",
        }])
        quality = {"a": {
            "carrier_qualification": "QUALIFIED", "carrier_tier": "CORE",
            "gate_failures": "",
        }}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _write_event_cache(event, ["a"], summary, pd.DataFrame(), root)

            cached = _load_event_cache(event, ["a"], quality, root)
            wrong_members = _load_event_cache(event, ["a", "b"], quality, root)
            wrong_ignition = _load_event_cache(
                {**event, "limit_stocks": "b"}, ["a"], quality, root
            )

        self.assertIsNotNone(cached)
        cached_summary, cached_trace = cached
        self.assertEqual(cached_summary.iloc[0]["current_qualification"], "QUALIFIED")
        self.assertTrue(cached_trace.empty)
        self.assertIsNone(wrong_members)
        self.assertIsNone(wrong_ignition)

    def test_report_cache_survives_daily_mapping_and_auxiliary_changes(self) -> None:
        calendar = ["2026-09-03", "2026-09-04"]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report = root / "20260903" / "weekly_reverse_trace"
            report.mkdir(parents=True)
            payload = {
                "version": VERSION,
                "carrier_version": CQ.VERSION,
                "replay_cache_version": REPLAY_CACHE_VERSION,
                "mapping": {"version": "old-mapping", "generated": "2026-09-03"},
                "semantic_auxiliary": {"updated": "old-aux"},
            }
            (report / "weekly_reverse_trace_summary.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            pd.DataFrame([{
                "raw_theme": "raw", "theme": "resolved", "t0": "2026-08-31",
                "symbol": "a",
            }]).to_csv(
                report / "pre_ignition_carriers.csv", index=False, encoding="utf-8-sig"
            )
            pd.DataFrame([{
                "raw_theme": "raw", "t0": "2026-08-31", "symbol": "a",
            }]).to_csv(
                report / "pre_ignition_daily_trace.csv", index=False, encoding="utf-8-sig"
            )

            source = _load_replay_source(
                calendar, "2026-09-04", "2026-09-04",
                {"version": "new-mapping"}, {"updated": "new-aux"}, root,
            )

        self.assertIsNotNone(source)
        self.assertEqual(source["path"], str(report))

    def test_daily_mapping_date_and_output_only_carrier_change_keep_cache_compatible(self) -> None:
        self.assertIn("carrier_quality_v1_1_20260829", REPLAY_COMPATIBLE_CARRIER_VERSIONS)
        self.assertIn(CQ.VERSION, REPLAY_COMPATIBLE_CARRIER_VERSIONS)
        previous = {
            "version": "weekly_reverse_trace_v1_2_20260901",
            "mapping": {"generated": "2026-09-02", "version": "theme_semantics_v1_2"},
        }
        current = {"generated": "2026-09-03", "version": "theme_semantics_v1_2"}
        self.assertNotEqual(previous["mapping"]["generated"], current["generated"])
        self.assertEqual(previous["mapping"]["version"], current["version"])

    def test_event_lifecycle_separates_new_reignition_and_expansion(self) -> None:
        calendar = [f"2026-08-{day:02d}" for day in range(1, 21)]
        panel = pd.DataFrame([
            {
                "date": "2026-08-02", "theme": "old_cycle", "member_count": 60,
                "ignition_threshold": 3, "limit_count": 3,
                "limit_stocks": "a|b|c",
            },
            {
                "date": "2026-08-15", "theme": "old_cycle", "member_count": 60,
                "ignition_threshold": 3, "limit_count": 3,
                "limit_stocks": "d|e|f",
            },
            {
                "date": "2026-08-16", "theme": "old_cycle", "member_count": 60,
                "ignition_threshold": 3, "limit_count": 4,
                "limit_stocks": "d|e|f|g",
            },
            {
                "date": "2026-08-17", "theme": "old_cycle", "member_count": 60,
                "ignition_threshold": 3, "limit_count": 3,
                "limit_stocks": "d|e|f",
            },
            {
                "date": "2026-08-15", "theme": "new_cycle", "member_count": 40,
                "ignition_threshold": 3, "limit_count": 3,
                "limit_stocks": "h|i|j",
            },
        ])

        events = detect_events(panel, calendar, "2026-08-15", "2026-08-17")
        keyed = {
            (row.theme, row.date): row
            for row in events.itertuples(index=False)
        }

        self.assertEqual(keyed[("old_cycle", "2026-08-15")].event_type, "REIGNITION")
        self.assertEqual(keyed[("old_cycle", "2026-08-15")].follow_through, "EXPANDED")
        self.assertEqual(keyed[("old_cycle", "2026-08-16")].event_type, "EXPANSION")
        self.assertEqual(keyed[("old_cycle", "2026-08-17")].event_type, "CONTINUATION")
        self.assertEqual(keyed[("new_cycle", "2026-08-15")].event_type,
                         "NEW_IGNITION_30D")
        self.assertEqual(keyed[("new_cycle", "2026-08-15")].follow_through, "FADED")


if __name__ == "__main__":
    unittest.main()

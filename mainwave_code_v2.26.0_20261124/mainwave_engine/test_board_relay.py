from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from . import board_auction as A, board_relay as W, eod_review as E
from .intraday_quotes import ApiError


DAY, NEXT = "2026-11-23", "2026-11-24"
S, T, H = "sh_600001", "sz_000002", "sh_600003"


def fixture():
    cut = W.clock(DAY, "20:00")
    facts = {"boards": {"U": [S, T, H], "Z": []}, "ladder": {"heights": {S: 1, T: 1, H: 5}}}
    attrs = {"rows": [{"symbol": s, "status": "CORROBORATED", "primary_theme": "Example", "name": s}
                      for s in (S, T, H)]}
    quotes = {s: {"amount": 1000000, "close": 10., "low": 9., "position60": .2, "one_price": False,
                  "gap_intraday_low_pct": -2} for s in (S, T, H)}
    refs = {S: {"first_time": "09:50:00", "last_time": "100000", "open_times": 1}, T: {}}
    return cut, facts, attrs, quotes, refs


def watch():
    cutoff, facts, attrs, quotes, refs = fixture()
    return W.compile_watch(DAY, facts, attrs, refs, quotes, None, cutoff)


class RelayTest(unittest.TestCase):
    def test_first_board_has_own_lanes_and_peer_excludes_self(self):
        report = watch()
        self.assertEqual(report["focus"], {"FIRST": [S], "RELAY": [], "HEIGHT": [H]})
        row = next(r for r in report["rows"] if r["symbol"] == S)
        self.assertEqual(set(row["peer_symbols"]), {T, H})
        self.assertEqual(row["peer_ladder"], {"1": 1, "5": 1})
        self.assertFalse(report["orders_enabled"])

    def test_reference_conflict_and_one_price_do_not_enter_early_focus(self):
        cut, facts, attrs, quotes, refs = fixture()
        facts["boards"]["reconciliation"] = {"U": {"local_only": [S]}}
        quotes[T]["one_price"] = True
        result = W.compile_watch(DAY, facts, attrs, refs, quotes, None, cut)
        self.assertEqual(result["focus"]["FIRST"], [])
        self.assertEqual({r["symbol"]: r["state"] for r in result["rows"]}[T], "ONE_PRICE_UNPROVEN")

    def test_seal_formats_and_invalid_values(self):
        self.assertEqual([W.seal_time(x) for x in ("093000", "09:30:00", 93000)], [34200] * 3)
        self.assertEqual([W.seal_time(x) for x in (None, "nan", "103099", 92000, 150001)], [None] * 5)

    def test_seat_duplicate_and_windows_do_not_sum(self):
        seats = [
            {"side": 0, "reason": "single", "exalter": "东方财富拉萨东路", "buy": 2, "net_buy": 1},
            {"side": 0, "reason": "single", "exalter": "东方财富拉萨东路", "buy": 2, "net_buy": 1},
            {"side": 0, "reason": "single", "exalter": "Other", "buy": 6, "net_buy": 4},
            {"side": 0, "reason": "multi", "exalter": "东方财富拉萨东路", "buy": 15, "net_buy": 5},
        ]
        out = {r["reason"]: r for r in W.seat_summary(seats, 100)["disclosures"]}
        self.assertEqual((out["single"]["seat_count"], out["single"]["buy_share"], out["single"]["net_to_day_amount"]),
                         (1, .25, .01))
        self.assertEqual(out["multi"]["buy_share"], 1.)

    def test_risk_is_time_bounded_and_keeps_height_visible(self):
        cut, facts, attrs, quotes, refs = fixture()
        title = H + "收到监管工作函"
        attrs["materials"] = {H: [{"kind": "MEDIA_BODY", "title": title, "published_at": cut.isoformat(),
                                    "collected_at": cut.isoformat()}],
                              S: [{"kind": "ISSUER_TITLE", "title": "监管工作函", "published_at": W.clock(DAY, "21:00").isoformat(),
                                   "collected_at": W.clock(DAY, "21:00").isoformat()}]}
        out = W.compile_watch(DAY, facts, attrs, refs, quotes, None, cut)
        states = {r["symbol"]: r["state"] for r in out["rows"]}
        self.assertEqual(states[H], "REGULATORY_REVIEW")
        self.assertEqual(states[S], "OBSERVATION")
        self.assertEqual(out["focus"]["HEIGHT"], [H])


class AuctionTest(unittest.TestCase):
    def fake(self, price=10.3):
        return pd.DataFrame([
            {"ts_code": "600001.SH", "trade_date": "20261124", "vol": 7500,
             "price": price, "amount": price * 7500, "pre_close": 10.},
            {"ts_code": "000002.SZ", "trade_date": "20261124", "vol": 5000,
             "price": 10.1, "amount": 50500, "pre_close": 10.},
        ])

    def test_confirm_requires_notices_peer_and_price(self):
        q = A.snapshot_rows(self.fake(), NEXT)
        self.assertEqual(A.compare(watch(), q)[0]["state"], "NOTICES_DATA_WAIT")
        n = {S: {"status": "AVAILABLE", "notices": []}}
        self.assertEqual(A.compare(watch(), q, n)[0]["state"], "WEAK_TO_STRONG_WATCH")
        self.assertEqual(A.compare(watch(), {S: q[S]}, n)[0]["state"], "PEERS_DATA_WAIT")
        q = A.snapshot_rows(self.fake(11), NEXT)
        self.assertEqual(A.compare(watch(), q, n)[0]["state"], "AT_LIMIT_FILL_UNPROVEN")
        n[S]["notices"] = [{"kind": "REGULATORY_REVIEW"}]
        self.assertEqual(A.compare(watch(), q, n)[0]["state"], "REGULATORY_REVIEW")

    def test_schema_and_amount_units_and_duplicate_dates(self):
        f = self.fake()
        self.assertEqual(A.snapshot_rows(f, DAY), {})
        f.loc[0, "amount"] *= 100
        self.assertNotIn(S, A.snapshot_rows(f, NEXT))
        self.assertEqual(A.snapshot_rows(pd.concat([f, f]), NEXT), {})

    def test_replay_cannot_be_used_and_late_data_cannot_confirm(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            (data_root / "000300_SH_index.csv").write_text(f"trade_date\n{DAY}\n", encoding="ascii")
            out = {**watch(), "status": "READY", "target": NEXT, "mode": "REPLAY",
                   "published_at": W.clock(DAY, "23:00").isoformat()}
            file = W.folder(root, DAY) / "forward/001.json"
            E._write(file, out)
            self.assertFalse(A.frozen(root, DAY, NEXT)[0])
            E._write(file, {**out, "mode": "FORWARD"})
            self.assertEqual(A.frozen(root, DAY, NEXT)[1], file)
            outer = self

            class Client:
                def query(self, api, **kwargs):
                    if api == "trade_cal":
                        return pd.DataFrame([{"cal_date": "20261124", "is_open": 1, "pretrade_date": "20261123"}])
                    return outer.fake()

            now = iter([W.clock(NEXT, "09:26"), W.clock(NEXT, "09:30")])
            report = A.run(NEXT, root, data_root, Client(), now=lambda: next(now))
            self.assertEqual(report["status"], "LATE_DATA")
            self.assertFalse(report["orders_enabled"])

            class Denied(Client):
                def query(self, api, **kwargs):
                    if api == "stk_auction":
                        raise ApiError(api, 40203, "denied")
                    return super().query(api, **kwargs)

            report = A.run(NEXT, root, data_root, Denied(), now=lambda: W.clock(NEXT, "09:26"))
            self.assertEqual(report["status"], "PERMISSION_DENIED")
            self.assertEqual(report["rows"][0]["state"], "DATA_WAIT")

    def test_holiday_forecast_cannot_include_late_reconstruction(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            actual = "2026-11-25"
            out = {**watch(), "status": "READY", "target": NEXT, "mode": "FORWARD",
                   "target_basis": "PROVISIONAL_WEEKDAY", "published_at": W.clock(DAY, "23:00").isoformat()}
            file = W.folder(root, DAY) / "forward/001.json"
            E._write(file, out)
            self.assertTrue(A.frozen(root, DAY, actual)[0])
            E._write(file, {**out, "published_at": W.clock(NEXT, "10:00").isoformat()})
            self.assertFalse(A.frozen(root, DAY, actual)[0])

    def test_api_keeps_replay_separate_from_auction(self):
        from flask import Flask
        from .trading import web

        app = Flask(__name__)
        app.register_blueprint(web.bp)
        with TemporaryDirectory() as tmp, patch.object(web.store, "ROOT", Path(tmp)):
            root = Path(tmp) / "results/mainwave_engine"
            payload = {**watch(), "target": NEXT, "mode": "FORWARD"}
            E._write(W.folder(root, NEXT) / "auction.json", {"as_of": NEXT, "previous_day": DAY, "status": "PERMISSION_DENIED"})
            E._write(W.folder(root, DAY) / "replay.json", {**payload, "mode": "REPLAY"})
            client = app.test_client()
            self.assertIsNone(client.get(f"/api/trading/board-relay?as_of={DAY}").json["auction"])
            E._write(W.folder(root, DAY) / "latest.json", payload)
            self.assertEqual(client.get(f"/api/trading/board-relay?as_of={DAY}").json["auction"]["status"], "PERMISSION_DENIED")


if __name__ == "__main__":
    unittest.main()

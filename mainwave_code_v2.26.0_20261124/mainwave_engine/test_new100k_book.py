"""NEW100K compounding book: own ledger, 2 stock seats, compounding risk base under a ceiling, drawdown pause,
effective date; plus the next-session check of B1/B3 shadow drafts."""
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import entry_contracts as EC
from .trading import books, paper, store
from .test_trading_system import AT, CONFIG, DAY, payload


class Base(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "data").mkdir()
        self.db = self.root / "t.sqlite3"
        self.payload = payload()
        with patch.object(store, "now", return_value=AT):
            self.snap = store.persist(self.payload, self.db)

    def tearDown(self):
        self.temp.cleanup()

    def run_book(self, **kw):
        with patch.object(store, "now", return_value=AT), patch.dict(books.NEW100K_PROTOCOL, {"effective_at": DAY}):
            return books.run_new100k(DAY, self.payload, self.snap, self.root, self.root / "data", **kw)


class LiveWindowTests(unittest.TestCase):
    def test_a_closure_past_midnight_is_still_live_and_a_later_rerun_is_not(self):
        from .trading.daily import live_run
        self.assertTrue(live_run("2026-09-22", "2026-09-22T21:00:00+08:00"))
        self.assertTrue(live_run("2026-09-22", "2026-09-23T00:10:00+08:00"), "跨午夜的收盘闭环仍是当日")
        self.assertFalse(live_run("2026-09-22", "2026-09-23T09:00:00+08:00"), "下一交易日开盘前窗口关闭")
        self.assertFalse(live_run("2026-09-22", "2026-09-22T14:59:00+08:00"))
        self.assertTrue(live_run("2026-09-25", "2026-09-27T20:00:00+08:00"), "周五收盘可在周末补跑")
        self.assertFalse(live_run("2026-09-25", "2026-09-28T09:00:00+08:00"))


class BookTests(Base):
    def test_not_effective_before_the_protocol_date(self):
        with patch.object(store, "now", return_value=AT):
            out = books.run_new100k(DAY, self.payload, self.snap, self.root, self.root / "data", live=True)
        self.assertEqual(out["status"], "NOT_YET_EFFECTIVE")
        self.assertFalse((self.root / books.NEW100K_DB).exists())

    def test_live_run_fills_two_stock_seats_in_its_own_ledger(self):
        out = self.run_book(live=True)
        statuses = [p["status"] for p in out["placed"]]
        self.assertEqual(statuses[:2], ["ORDER_APPROVED", "ORDER_APPROVED"], out["placed"])
        self.assertEqual(statuses[2], "NOT_CREATED")
        self.assertIn("股票席位已满", out["placed"][2]["detail"])
        self.assertTrue(out["account_matches_code"])
        db = self.root / books.NEW100K_DB
        self.assertNotEqual(db, self.root / books.AUTO_DB)
        with paper.database(db) as conn:
            plans = [json.loads(r[0]) for r in conn.execute("SELECT payload FROM paper_plans")]
            _, cfg = paper._account(conn)
        self.assertEqual({p["book"] for p in plans}, {"NEW100K"})
        self.assertEqual((cfg["capital_policy"], cfg["stock_seats"], cfg["equity_ceiling_cents"]), ("COMPOUND_CEILING", 2, 10000000))
        self.assertFalse((self.root / books.AUTO_DB).exists(), "复利账不得触碰旧自动账")

    def test_reconstruction_places_nothing(self):
        out = self.run_book(live=False)
        self.assertEqual({p["status"] for p in out["placed"]}, {"NOT_LIVE_DAY_NO_ORDER"})
        self.assertFalse(out["ledger_touched"])

    def test_drawdown_pause_blocks_new_entries(self):
        with patch.object(books, "current_drawdown", return_value=12.0):
            out = self.run_book(live=True)
        self.assertTrue(out["paused"])
        self.assertEqual({p["status"] for p in out["placed"]}, {"DRAWDOWN_PAUSE"})

    def test_current_drawdown_reads_the_marked_nav(self):
        self.assertIsNone(books.current_drawdown({"nav": [{"equity_cents": None}]}))
        nav = [{"equity_cents": 100}, {"equity_cents": 120}, {"equity_cents": None}, {"equity_cents": 108}]
        self.assertEqual(books.current_drawdown({"nav": nav}), 10.0)


class LedgerProtocolTests(Base):
    def body(self, **extra):
        return {**CONFIG, **extra}

    def test_optional_keys_are_validated_and_absent_keys_keep_old_behaviour(self):
        with self.assertRaises(ValueError):
            paper.initialize(self.body(capital_policy="GROW_FAST"), self.root / "a.sqlite3")
        with self.assertRaises(ValueError):
            paper.initialize(self.body(capital_policy="COMPOUND_CEILING", equity_ceiling=50000), self.root / "b.sqlite3")
        with self.assertRaises(ValueError):
            paper.initialize(self.body(stock_seats=0), self.root / "c.sqlite3")
        paper.initialize(CONFIG, self.root / "d.sqlite3")
        with paper.database(self.root / "d.sqlite3") as conn:
            _, cfg = paper._account(conn)
        self.assertNotIn("capital_policy", cfg)

    def quantity_after_gain(self, **extra):
        db = self.root / f"{extra.get('capital_policy', 'x')}_{extra.get('equity_ceiling', 0)}.sqlite3"
        with patch.object(store, "now", return_value=AT):
            store.persist(self.payload, db)
            paper.initialize(self.body(**extra), db, at=AT)
            with sqlite3.connect(db) as conn:
                conn.execute("UPDATE paper_account SET cash_cents=?", (15000000,))  # equity 150,000 after gains
            snap = store.persist(self.payload, db)
            paper.submit(snap["snapshot_id"], "sh_600013", db, at=AT)
        with paper.database(db) as conn:
            return json.loads(conn.execute("SELECT payload FROM paper_plans").fetchone()[0])["quantity"]

    def test_compounding_base_grows_only_up_to_the_ceiling(self):
        frozen = self.quantity_after_gain()
        capped = self.quantity_after_gain(capital_policy="COMPOUND_CEILING", equity_ceiling=100000)
        grown = self.quantity_after_gain(capital_policy="COMPOUND_CEILING", equity_ceiling=200000)
        self.assertEqual(frozen, capped)
        self.assertGreater(grown, capped)


class DraftCheckTests(unittest.TestCase):
    def test_contract_reading_of_next_session(self):
        drafts = [{"symbol": "a", "contract": "B1", "effective_trade_date": "2026-09-23", "entry_max": 10.2, "invalidation": 9.4, "ready": True},
                  {"symbol": "b", "contract": "B1", "effective_trade_date": "2026-09-23", "entry_max": 10.2, "invalidation": 9.4, "ready": True},
                  {"symbol": "c", "contract": "B3", "effective_trade_date": "2026-09-23", "entry_max": 10.2, "invalidation": 9.4, "ready": True},
                  {"symbol": "d", "contract": "B3", "effective_trade_date": "2026-09-23", "entry_max": 10.2, "invalidation": 9.4, "ready": True},
                  {"symbol": "e", "contract": "B1", "effective_trade_date": "2026-09-24", "entry_max": 10.2, "invalidation": 9.4, "ready": True}]
        bars = {"a": {"open": 10.0, "close": 10.5}, "b": {"open": 10.8, "close": 11.0}, "c": {"open": 9.3, "close": 9.0}}
        out = EC.forward_check(drafts, bars, "2026-09-23")
        by = {r["symbol"]: r for r in out["rows"]}
        self.assertEqual((by["a"]["result"], by["b"]["result"], by["c"]["result"], by["d"]["result"]),
                         ("OPEN_IN_BAND", "GAP_ABOVE_BAND_NO_CHASE", "OPEN_AT_OR_BELOW_INVALIDATION", "PENDING"))
        self.assertNotIn("e", by)  # a draft for another session is not checked today
        self.assertTrue(by["a"]["close_held_invalidation"])
        self.assertEqual((out["checked"], out["in_band"], out["in_band_close_held"]), (4, 1, 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""strong_trend_core_v1 regressions: campaign profit accounting, the independent position monitor, and the
STRONG_TREND shadow entries/exit. Numbers follow the design package's synthetic case (§02.8)."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from . import strong_trend as ST
from .trading import paper, position_monitor as PM, profit_tracker as PT, store

CFG = {"fee_bps": 0.0, "min_fee_cents": 0, "sell_tax_bps": 0.0, "capital_policy": "INITIAL_CAP"}


class Ledger:
    """A minimal paper ledger written straight into the tables the gateway uses."""

    def __init__(self, db, capital=10000000, cfg=CFG):
        self.db, self.n = db, 0
        with paper.database(db) as conn:
            conn.execute("INSERT INTO snapshots VALUES ('S','2026-09-01','2026-09-01','{}')")
            conn.execute("INSERT INTO paper_account VALUES (1,?,?,?,?)", (capital, capital, "2026-09-01", json.dumps(cfg)))

    def plan(self, pid="P1", symbol="sh_600001", stop=900, **extra):
        payload = {"id": pid, "symbol": symbol, "name": "测试", "theme": "T", "book": "MANUAL", "mode": "TREND_PULLBACK",
                   "setup_kind": "FIRST_PULLBACK", "stop_cents": stop, "exit_version": "old_exit", **extra}
        with paper.database(self.db) as conn:
            conn.execute("INSERT INTO paper_plans VALUES (?,?,?,?,?)", (pid, "S", "O" + pid, "2026-09-01", json.dumps(payload)))
        return payload

    def fill(self, pid, side, day, qty, price, fee, status="FILLED"):
        self.n += 1
        oid, fid = f"{pid}:o{self.n}", f"{pid}:f{self.n}"
        cents = round(price * 100)
        with paper.database(self.db) as conn:
            conn.execute("INSERT INTO paper_orders VALUES (?,?,?,?,?,?,?,?,?,?)",
                         (oid, pid, side, qty, qty, status, 0, 0, day + "T09:00:00+08:00", "{}"))
            conn.execute("INSERT INTO paper_fills VALUES (?,?,?,?,?,?)", (fid, oid, day, qty, cents, fee))
            if side == "BUY":
                conn.execute("INSERT INTO paper_lots VALUES (?,?,?,?,?,?,?,?)", (fid, pid, "sh_600001", day, qty, qty, cents, fee))
                conn.execute("UPDATE paper_account SET cash_cents=cash_cents-?", (qty * cents + fee,))
            else:
                left = qty
                for lot in conn.execute("SELECT id, remaining FROM paper_lots WHERE plan_id=? AND remaining>0 ORDER BY entry_day, id", (pid,)).fetchall():
                    used = min(left, lot["remaining"])
                    conn.execute("UPDATE paper_lots SET remaining=remaining-? WHERE id=?", (used, lot["id"]))
                    left -= used
                conn.execute("UPDATE paper_account SET cash_cents=cash_cents+?", (qty * cents - fee,))
        return oid

    def order(self, pid, side, qty, status, day="2026-09-10"):
        with paper.database(self.db) as conn:
            conn.execute("INSERT INTO paper_orders VALUES (?,?,?,?,?,?,?,?,?,?)",
                         (f"{pid}:{status}", pid, side, qty, 0, status, 0, 0, day + "T15:00:00+08:00", "{}"))

    def value(self, pid, day, mark, status="HOLD", **floors):
        body = {"day": day, "status": status, "mark_cents": None if mark is None else round(mark * 100), **floors}
        with paper.database(self.db) as conn:
            conn.execute("INSERT INTO paper_valuations VALUES (?,?,?,?)", (f"{pid}:{day}", day, pid, json.dumps(body)))

    def campaign(self, plan, day, protection=None):
        with paper.database(self.db) as conn:
            return PT.campaign(conn, plan, day, CFG, protection)


class Temp(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.book = Ledger(self.root / "book.sqlite3")

    def tearDown(self):
        self.temp.cleanup()


class ProfitTests(Temp):
    def design_case(self):
        plan = self.book.plan()
        self.book.fill("P1", "BUY", "2026-09-02", 1000, 10.00, 2000)
        self.book.fill("P1", "BUY", "2026-09-03", 500, 11.00, 1500)
        self.book.value("P1", "2026-09-04", 12.50)
        return plan

    def test_design_case_add_partial_sell_giveback_and_scenarios(self):
        plan = self.design_case()
        self.assertEqual(self.book.campaign(plan, "2026-09-04")["net_pnl_cents"], 321500)
        self.book.fill("P1", "SELL", "2026-09-05", 300, 12.50, 1000)
        self.book.value("P1", "2026-09-05", 12.50)
        self.assertEqual(self.book.campaign(plan, "2026-09-05")["net_pnl_cents"], 320500, "部分卖出只减少卖出费用")
        self.book.value("P1", "2026-09-06", 12.00)
        c = self.book.campaign(plan, "2026-09-06", protection=1080)
        self.assertEqual((c["net_pnl_cents"], c["observed_peak_cents"], c["giveback_cents"]), (260500, 321500, 61000))
        self.assertAlmostEqual(c["giveback_ratio"], 0.189736, places=5)
        self.assertEqual((c["realized_fifo_cents"], c["unrealized_fifo_cents"]), (73400, 187100))
        self.assertTrue(c["reconciled"])
        # Zero-fee config: the design's 20-yuan exit cost is replaced by the configured fee (0 here).
        self.assertEqual(c["net_pnl_if_exit_at_protection_cents"], 260500 + 1200 * (1080 - 1200))
        self.assertEqual(c["current_wealth_loss_in_stress_cents"], c["net_pnl_cents"] - c["net_pnl_if_stress_exit_cents"])
        self.assertIn("非锁定利润", c["projection_warning"])

    def test_peak_uses_the_quantity_held_on_that_day(self):
        plan = self.book.plan()
        self.book.fill("P1", "BUY", "2026-09-02", 100, 10.00, 0)
        self.book.value("P1", "2026-09-02", 20.00)
        self.book.fill("P1", "BUY", "2026-09-03", 1000, 12.00, 0)
        self.book.value("P1", "2026-09-03", 12.00)
        c = self.book.campaign(plan, "2026-09-03")
        self.assertEqual(c["observed_peak_cents"], 100000, "旧峰值只按当时 100 股计算，不按今天 1100 股回算")
        self.assertEqual(c["net_pnl_cents"], 20000)

    def test_giveback_can_exceed_one_hundred_percent(self):
        plan = self.book.plan()
        self.book.fill("P1", "BUY", "2026-09-02", 1000, 10.00, 0)
        self.book.value("P1", "2026-09-03", 11.00)
        self.book.value("P1", "2026-09-04", 9.50)
        c = self.book.campaign(plan, "2026-09-04")
        self.assertEqual(c["giveback_ratio"], 1.5)
        self.assertTrue(c["giveback_alert"])

    def test_no_positive_peak_reports_no_ratio(self):
        plan = self.book.plan()
        self.book.fill("P1", "BUY", "2026-09-02", 1000, 10.00, 0)
        self.book.value("P1", "2026-09-03", 9.00)
        self.assertIsNone(self.book.campaign(plan, "2026-09-03")["giveback_ratio"])

    def test_data_wait_mark_is_not_zero_and_not_a_peak(self):
        plan = self.book.plan()
        self.book.fill("P1", "BUY", "2026-09-02", 1000, 10.00, 0)
        self.book.value("P1", "2026-09-03", 11.00)
        self.book.value("P1", "2026-09-04", None, status="DATA_WAIT")
        c = self.book.campaign(plan, "2026-09-04")
        self.assertEqual((c["mark_day"], c["net_pnl_cents"], c["observed_peak_cents"]), ("2026-09-03", 100000, 100000))

    def test_closed_campaign_is_frozen(self):
        plan = self.book.plan()
        self.book.fill("P1", "BUY", "2026-09-02", 1000, 10.00, 500)
        self.book.value("P1", "2026-09-03", 12.00)
        self.book.fill("P1", "SELL", "2026-09-04", 1000, 11.00, 500)
        c = self.book.campaign(plan, "2026-09-30")
        self.assertEqual((c["trade_closed"], c["projection_status"], c["net_pnl_cents"], c["observed_peak_cents"]),
                         (True, "CLOSED_FROZEN", 99000, 199500))

    def test_external_cash_is_not_return_and_does_not_reset_drawdown(self):
        series = [{"day": "d1", "equity_cents": 10000000}, {"day": "d2", "equity_cents": 11000000},
                  {"day": "d3", "equity_cents": 16000000}, {"day": "d4", "equity_cents": 17600000}]
        rows, worst = PT.unit_nav(series, 10000000, {"d3": 5000000})
        self.assertAlmostEqual(rows[-1]["unit_nav"], 1.21, places=6)
        self.assertEqual(rows[2]["unit_nav"], 1.1, "充值当天单位净值不跳升")
        self.assertAlmostEqual(PT.time_weighted_return([(10000000, 11000000), (16000000, 17600000)]), 0.21)
        rows, worst = PT.unit_nav([{"day": "d1", "equity_cents": 10000000}, {"day": "d2", "equity_cents": 8000000},
                                   {"day": "d3", "equity_cents": 13000000}], 10000000, {"d3": 5000000})
        self.assertEqual((rows[-1]["unit_nav"], worst), (0.8, -0.2), "亏损后充值使金额创新高，但单位净值回撤不重置")
        with self.assertRaises(ValueError):
            PT.time_weighted_return([])


def write_bars(root, symbol, rows):
    path = Path(root) / f"{symbol}_daily.csv"
    pd.DataFrame(rows).to_csv(path, index=False)


def bars(day, close, low=None, status=1):
    return {"date": day, "open": close, "high": close, "low": low if low is not None else close, "close": close,
            "preclose": close, "volume": 1000, "tradestatus": status, "isST": 0}


class MonitorTests(Temp):
    def run_monitor(self, day="2026-09-10"):
        out = PM.run(day, data_root=self.root, ledgers={"MANUAL": self.book.db, "ABSENT": self.root / "none.sqlite3"})
        return out, out["books"]["MANUAL"]["positions"]

    def test_runs_without_any_pipeline_output_and_never_creates_a_missing_ledger(self):
        self.book.plan()
        self.book.fill("P1", "BUY", "2026-09-02", 1000, 10.00, 0)
        write_bars(self.root, "sh_600001", [bars("2026-09-09", 10.5), bars("2026-09-10", 10.8)])
        out, pos = self.run_monitor()
        self.assertEqual(out["books"]["ABSENT"]["status"], "NO_LEDGER")
        self.assertFalse((self.root / "none.sqlite3").exists())
        self.assertEqual((pos[0]["data_state"], pos[0]["action"], pos[0]["profit"]["net_pnl_cents"]), ("FRESH", "HOLD", 80000))
        self.assertTrue(out["books"]["MANUAL"]["reconciled"])

    def test_missing_bar_is_manual_review_not_normal(self):
        self.book.plan()
        self.book.fill("P1", "BUY", "2026-09-02", 1000, 10.00, 0)
        write_bars(self.root, "sh_600001", [bars("2026-09-08", 10.5)])
        _, pos = self.run_monitor()
        self.assertEqual((pos[0]["data_state"], pos[0]["action"]), ("STALE", "MANUAL_REVIEW"))
        self.assertIn("数据未就绪", pos[0]["add_blocked_by"])

    def test_close_below_the_protection_already_in_force_exits(self):
        self.book.plan(stop=800)
        self.book.fill("P1", "BUY", "2026-09-02", 1000, 10.00, 0)
        self.book.value("P1", "2026-09-09", 10.50, soft_floor=10.20, hard_floor=8.00)
        write_bars(self.root, "sh_600001", [bars("2026-09-10", 10.10, low=10.00)])
        _, pos = self.run_monitor()
        self.assertEqual((pos[0]["trend_state"], pos[0]["action"]), ("BROKEN", "EXIT_REQUIRED"))

    def test_todays_new_protection_does_not_apply_to_today(self):
        self.book.plan(stop=800)
        self.book.fill("P1", "BUY", "2026-09-02", 1000, 10.00, 0)
        self.book.value("P1", "2026-09-09", 10.10, soft_floor=9.00, hard_floor=8.00)
        self.book.value("P1", "2026-09-10", 10.10, soft_floor=10.50, hard_floor=8.00)
        write_bars(self.root, "sh_600001", [bars("2026-09-10", 10.10, low=10.00)])
        _, pos = self.run_monitor()
        q = pos[0]["six_questions"]["protection"]
        self.assertEqual((q["effective_today_cents"], q["next_session_cents"]), (900, 1050))
        self.assertEqual(pos[0]["action"], "HOLD", "今日收盘生成的新保护线只从下一交易日生效")

    def test_blocked_exit_stays_an_exit_after_a_rebound(self):
        self.book.plan()
        self.book.fill("P1", "BUY", "2026-09-02", 1000, 10.00, 0)
        self.book.order("P1", "SELL", 1000, "EXIT_BLOCKED")
        write_bars(self.root, "sh_600001", [bars("2026-09-10", 12.00)])
        out, pos = self.run_monitor()
        self.assertEqual((pos[0]["trade_state"], pos[0]["action"]), ("EXIT_BLOCKED", "EXIT_REQUIRED"))
        self.assertEqual(out["alerts"][0]["level"], "P0")
        self.assertIn("存在退出/减仓要求", pos[0]["add_blocked_by"])

    def test_new_lot_is_locked_by_t_plus_1(self):
        self.book.plan()
        self.book.fill("P1", "BUY", "2026-09-09", 1000, 10.00, 0)
        self.book.fill("P1", "BUY", "2026-09-10", 500, 10.00, 0)
        write_bars(self.root, "sh_600001", [bars("2026-09-10", 10.00)])
        _, pos = self.run_monitor()
        self.assertEqual((pos[0]["execution_state"]["sellable"], pos[0]["execution_state"]["t_plus_1_locked"]), (1000, 500))


def write_index(root, closes, end="2026-09-10"):
    days = pd.bdate_range(end=end, periods=len(closes)).strftime("%Y%m%d")
    pd.DataFrame({"ts_code": "000852.SH", "trade_date": days, "close": closes}).to_csv(Path(root) / "000852_SH_index.csv", index=False)


class MarketRiskTests(Temp):
    def test_trend_exit_is_alert_only_and_rendered_red(self):
        self.book.plan()
        self.book.fill("P1", "BUY", "2026-09-02", 1000, 10.00, 0)
        write_bars(self.root, "sh_600001", [bars("2026-09-10", 10.8)])
        write_index(self.root, [100 - i * 0.5 for i in range(30)])
        out = PM.run("2026-09-10", data_root=self.root, ledgers={"MANUAL": self.book.db})
        self.assertEqual(out["market_risk"]["level"], "TREND_EXIT")
        self.assertEqual(out["books"]["MANUAL"]["positions"][0]["action"], "HOLD", "大盘提示不改变持仓动作")
        self.assertEqual(out["status"], "READY")
        self.assertIn("background:#c00000", PM.render(out))

    def test_critical_is_rendered_orange_and_changes_no_action(self):
        self.book.plan()
        self.book.fill("P1", "BUY", "2026-09-02", 1000, 10.00, 0)
        write_bars(self.root, "sh_600001", [bars("2026-09-10", 10.8)])
        rising = [100 + 0.1 * i for i in range(29)]
        write_index(self.root, rising + [rising[-1] - 5])
        out = PM.run("2026-09-10", data_root=self.root, ledgers={"MANUAL": self.book.db})
        self.assertEqual(out["market_risk"]["level"], "CRITICAL")
        self.assertEqual(out["books"]["MANUAL"]["positions"][0]["action"], "HOLD")
        html = PM.render(out)
        self.assertIn("🟠 大盘趋势临界", html)
        self.assertNotIn("🟢", html)

    def test_levels(self):
        write_index(self.root, [100 + i * 0.5 for i in range(30)])
        self.assertEqual(PM.market_risk("2026-09-10", self.root)["level"], "NORMAL")
        rebound = [100 - i * 0.5 for i in range(29)] + [100.0]
        write_index(self.root, rebound)
        self.assertEqual(PM.market_risk("2026-09-10", self.root)["level"], "NORMAL", "收盘回到 MA20 上方即解除")
        rising = [100 + 0.1 * i for i in range(29)]
        write_index(self.root, rising + [rising[-1] - 5])
        r = PM.market_risk("2026-09-10", self.root)
        self.assertEqual(r["level"], "CRITICAL", "收盘在 MA20 下方，MA20 尚未下拐，但明日平收就会下拐")
        self.assertLess(r["ma20_if_flat_tomorrow"], r["ma20"])
        write_index(self.root, rising + [rising[-1] - 5] * 2)
        self.assertEqual(PM.market_risk("2026-09-10", self.root)["level"], "TREND_EXIT")
        self.assertEqual(PM.market_risk("2026-09-11", self.root)["level"], "STALE")
        self.assertEqual(PM.market_risk("2026-09-10", self.root / "absent")["level"], "UNKNOWN")


class MarketWarningTests(Temp):
    def test_warns_only_in_the_first_sessions_after_ma20_turns_down(self):
        write_index(self.root, [100.0] * 70)
        self.assertEqual(PM.market_warning("2026-09-10", self.root)["level"], "NORMAL")
        write_index(self.root, [100.0] * 69 + [90.0])
        w = PM.market_warning("2026-09-10", self.root)
        row = next(r for r in w["indices"] if r["index"] == "中证1000")
        self.assertEqual((w["level"], row["ma20_falling_days"], row["ma20_falling_since"]), ("WARNING", 1, "2026-09-10"))
        self.assertEqual(w["report_only"], True)
        write_index(self.root, [100.0] * 60 + [90.0] * 10)
        self.assertEqual(PM.market_warning("2026-09-10", self.root)["level"], "NORMAL", "MA20 下行超过 6 天不再预警")

    def test_rendered_and_does_not_change_positions(self):
        self.book.plan()
        self.book.fill("P1", "BUY", "2026-09-02", 1000, 10.00, 0)
        write_bars(self.root, "sh_600001", [bars("2026-09-10", 10.8)])
        write_index(self.root, [100.0] * 69 + [99.0])
        out = PM.run("2026-09-10", data_root=self.root, ledgers={"MANUAL": self.book.db})
        self.assertEqual(out["market_warning"]["level"], "WARNING")
        self.assertEqual(out["books"]["MANUAL"]["positions"][0]["action"], "HOLD")
        self.assertIn("短期回调预警", PM.render(out))


class ReboundBasketTests(Temp):
    def stocks(self, end="2026-09-10"):
        days = pd.bdate_range(end=end, periods=45).strftime("%Y-%m-%d")
        for i in range(5):
            pct = [-2.0 if i == 0 else 0.1] * len(days)  # sh_600010 is the most beaten
            pd.DataFrame({"date": days, "pct_chg": pct, "tradestatus": 1, "isST": 0, "amount": 1e8 * (i + 1)}).to_csv(
                self.root / f"sh_6000{i}0_daily.csv", index=False)

    def test_inactive_when_the_index_is_near_its_high(self):
        write_index(self.root, [100 + i * 0.1 for i in range(70)])
        self.assertEqual(PM.rebound_basket("2026-09-10", self.root)["level"], "INACTIVE")

    def test_first_minus_15_day_triggers_tranche_one_with_the_beaten_basket(self):
        self.stocks()
        write_index(self.root, [100.0] * 65 + [95, 92, 90, 88, 84])
        rb = PM.rebound_basket("2026-09-10", self.root)
        self.assertEqual((rb["level"], rb["tranches"][0]["signal_day"], rb["tranches"][1]["signal_day"]),
                         ("TRIGGERED_TODAY", "2026-09-10", None))
        self.assertEqual([p["symbol"] for p in rb["basket"]["picks"]], ["sh_600000"])
        out = PM.run("2026-09-10", data_root=self.root, ledgers={})
        self.assertIn("大跌反弹篮子触发", PM.render(out))
        self.assertEqual(out["status"], "READY", "反弹篮子只提示，不影响运行状态")

    def test_next_day_is_active_not_a_new_trigger(self):
        write_index(self.root, [100.0] * 65 + [95, 92, 90, 88, 84, 85], end="2026-09-11")
        rb = PM.rebound_basket("2026-09-11", self.root)
        self.assertEqual((rb["level"], rb["leg_start"]), ("ACTIVE", "2026-09-10"))


def frame(closes, highs=None, lows=None, start="2026-01-01"):
    days = pd.bdate_range(start, periods=len(closes)).strftime("%Y-%m-%d")
    closes = np.asarray(closes, float)
    highs = np.asarray(highs if highs is not None else closes * 1.01, float)
    lows = np.asarray(lows if lows is not None else closes * 0.99, float)
    pre = np.r_[closes[0], closes[:-1]]
    return pd.DataFrame({"date": days, "open": closes, "high": highs, "low": lows, "close": closes, "preclose": pre,
                         "volume": 1000.0, "tradestatus": 1, "isST": 0})


def bench_for(f):
    return pd.Series(np.full(len(f), 100.0), index=f["date"])


class StrongTrendTests(unittest.TestCase):
    def test_breakout_needs_no_pullback_and_excludes_the_signal_bar(self):
        f = frame([10.0] * 70 + [10.5])
        row = ST.evaluate(f, bench_for(f), f["date"].iloc[-1], "sh_600001", "测试")
        self.assertEqual((row["entry"], row["plan_legal"]), ("BREAKOUT", True))
        self.assertAlmostEqual(row["boundary"], 10.1, places=6)
        self.assertLess(row["invalidation"], row["band"][0])
        g = frame([10.0] * 70 + [10.05], highs=[10.1] * 70 + [11.0])
        self.assertIsNone(ST.evaluate(g, bench_for(g), g["date"].iloc[-1], "sh_600001"), "当日高点不进入 K_t")

    def test_reacceleration_after_a_pause_and_waiting_without_restrength(self):
        base = [10.0] * 60 + [10.8, 11.2, 11.0, 10.9, 11.0]
        f = frame(base + [11.5])
        row = ST.evaluate(f, bench_for(f), f["date"].iloc[-1], "sh_600001")
        self.assertEqual(row["entry"], "REACCELERATION")
        self.assertAlmostEqual(row["invalidation"], round(10.9 * 0.99 - 0.01, 4), places=3)
        g = frame(base + [11.05])
        self.assertIsNone(ST.evaluate(g, bench_for(g), g["date"].iloc[-1], "sh_600001"), "只有回落没有再转强：继续等待")

    def test_split_is_not_a_breakdown_or_breakout(self):
        f = frame([20.0] * 70 + [10.0])
        f.loc[70, "preclose"] = 10.0  # 2-for-1 on the last day, priced at the new basis
        self.assertIsNone(ST.evaluate(f, bench_for(f), f["date"].iloc[-1], "sh_600001"))
        self.assertAlmostEqual(ST.adjusted(f)["close"].iloc[0], 10.0)

    def test_each_bar_is_judged_by_the_protection_computed_through_the_previous_bar(self):
        closes = [10.0] * 20 + [11.0] * 20 + [10.4]
        lows = [9.5] * 20 + [10.5] * 20 + [10.3]
        f = frame(closes, lows=lows)
        out = ST.replay_exit(f, 20, hard=8.0)
        # Through bar 39 the channel is min(lows 20..39)=10.5, in force from bar 40; bar 39 itself was judged at 9.5.
        self.assertEqual((out["reason"], out["signal_index"], out["protection"]), ("TREND_EXIT", 40, 10.5))

    def test_close_below_protection_exits_at_the_next_open_not_at_the_line(self):
        f = frame([10.0] * 30, lows=[9.5] * 30)
        f.loc[28, ["close", "low"]] = [9.4, 9.3]
        f.loc[29, "open"] = 9.0
        out = ST.replay_exit(f, 20, hard=8.0)
        self.assertEqual((out["reason"], out["signal_index"], out["exit_price"]), ("TREND_EXIT", 28, 9.0))

    def test_protection_only_ratchets_up(self):
        broken, level = ST.protection_step(10.0, 11.0, [9.0] * 20)
        self.assertEqual((broken, level), (False, 10.0))
        self.assertEqual(ST.protection_step(10.0, 11.0, [9.0] * 19), (False, 10.0), "窗口不完整：保留已生效保护")


if __name__ == "__main__":
    unittest.main(verbosity=2)

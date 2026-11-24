"""Allocation: named accounts, CONFIG_PENDING blocks everything, seats by security, no cross-account approval."""
import unittest

from .trading import allocation as A


def plan(symbol, account="AUTO100W", date="2026-09-23", approval="APPROVED", theme="T", cap=10000, **over):
    p = {"plan_id": f"P-{symbol}", "account_id": account, "symbol": symbol, "approval_state": approval, "effective_trade_date": date,
         "entry_min": 10.0, "entry_max": 10.2, "invalidation": 9.5, "quantity_cap": cap, "planned_loss_budget": 1000.0, "primary_theme": theme}
    p.update(over)
    return p


ACTIVE = {"account_id": "AUTO100W", "status": "ACTIVE", "seats": {"stock": 2, "etf": 1}, "approved_equity_ceiling": None,
          "risk_profile": {"single_risk_pct": 1.0, "total_risk_pct": 3.0, "symbol_cap_pct": 20.0, "etf_cap_pct": 30.0, "theme_cap_pct": 40.0, "gross_cap_pct": 80.0}}


class AllocationTests(unittest.TestCase):
    def test_config_pending_blocks_everything(self):
        pending = dict(A.ACCOUNTS["NEW100K_PAPER"], risk_profile=dict(A.ACCOUNTS["NEW100K_PAPER"]["risk_profile"], max_drawdown_pct=None))
        out = A.allocate_ready_plans([plan("sh_600001", account="NEW100K_PAPER")], pending, "2026-09-24", cash=100000, verified_equity=100000)
        self.assertEqual(out["allocated"], [])
        self.assertEqual(out["blocked"][0]["reason"], A.CONFIG_PENDING)

    def test_new100k_protocol_matches_the_ledger_account(self):
        from .trading import books
        acct = A.ACCOUNTS["NEW100K_PAPER"]
        self.assertEqual((acct["status"], acct["seats"], acct["effective_at"]), ("ACTIVE", {"stock": 2, "etf": 1}, books.NEW100K_PROTOCOL["effective_at"]))
        for k in ("single_risk_pct", "total_risk_pct", "symbol_cap_pct", "theme_cap_pct", "gross_cap_pct"):
            self.assertEqual(float(acct["risk_profile"][k]), float(books.NEW100K_ACCOUNT[k]), k)
        self.assertEqual(acct["risk_profile"]["max_drawdown_pct"], books.NEW100K_PROTOCOL["drawdown_pause_pct"])
        self.assertEqual(acct["approved_equity_ceiling"], books.NEW100K_ACCOUNT["equity_ceiling"])
        out = A.allocate_ready_plans([plan("sh_600001", account="NEW100K_PAPER", date="2026-09-24")], acct, "2026-09-24", cash=100000, verified_equity=100000)
        self.assertEqual(len(out["allocated"]), 1)
        self.assertLessEqual(out["allocated"][0]["planned_loss"], 1000)

    def test_cross_account_expired_and_draft_are_named_blocks(self):
        plans = [plan("sh_600001", account="NEW100K_PAPER"), plan("sh_600002", date="2026-09-22"), plan("sh_600003", approval="DRAFT"),
                 plan("sh_600004", invalidation=None)]
        out = A.allocate_ready_plans(plans, ACTIVE, "2026-09-23", cash=1000000, verified_equity=1000000)
        reasons = {b["symbol"]: b["reason"] for b in out["blocked"]}
        self.assertEqual(reasons["sh_600001"], "ACCOUNT_MISMATCH:NEW100K_PAPER")
        self.assertEqual(reasons["sh_600002"], "EXPIRED_OR_NOT_YET_EFFECTIVE")
        self.assertEqual(reasons["sh_600003"], "NOT_APPROVED:DRAFT")
        self.assertTrue(reasons["sh_600004"].startswith("MISSING:invalidation"))
        self.assertEqual(out["allocated"], [])

    def test_seats_by_security_with_holdings_and_pending(self):
        plans = [plan("sh_600001", priority=1), plan("sh_600001", theme="OTHER", priority=2, plan_id="P-dup"), plan("sh_600003", priority=3), plan("sh_510300", priority=4)]
        holdings = [{"symbol": "sh_600009", "theme": "H", "cost": 50000, "risk": 500}]
        pending = [{"symbol": "sh_510050", "side": "BUY", "reserve": 20000, "risk": 300}]
        out = A.allocate_ready_plans(plans, ACTIVE, "2026-09-23", holdings, pending, cash=500000, verified_equity=1000000)
        allocated = [a["symbol"] for a in out["allocated"]]
        self.assertEqual(allocated, ["sh_600001"])  # one stock seat left; the ETF seat is held by the pending ETF order
        reasons = {(b["plan_id"], b["reason"]) for b in out["blocked"]}
        self.assertIn(("P-dup", "SECURITY_ALREADY_OCCUPIES_SEAT"), reasons)
        self.assertIn(("P-sh_600003", "NO_SEAT_STOCK"), reasons)
        self.assertIn(("P-sh_510300", "NO_SEAT_ETF"), reasons)
        a = out["allocated"][0]
        self.assertEqual(a["quantity"] % 100, 0)
        self.assertLessEqual(a["planned_loss"], 1000000 * 0.01)

    def test_risk_base_and_capital_block(self):
        self.assertEqual(A.risk_base(120000, 100000), 100000)
        self.assertEqual(A.risk_base(90000, 100000), 90000)
        self.assertEqual(A.risk_base(90000, None), 90000)
        self.assertIsNone(A.risk_base(None, 100000))
        tight = dict(ACTIVE, risk_profile=dict(ACTIVE["risk_profile"], single_risk_pct=0.001))
        out = A.allocate_ready_plans([plan("sh_600001")], tight, "2026-09-23", cash=1000000, verified_equity=1000000)
        self.assertEqual(out["blocked"][0]["reason"], "CAPITAL_BLOCK")
        out = A.allocate_ready_plans([plan("sh_600001")], ACTIVE, "2026-09-23", cash=None, verified_equity=1000000)
        self.assertEqual(out["blocked"][0]["reason"], "EQUITY_OR_CASH_UNVERIFIED")


if __name__ == "__main__":
    unittest.main(verbosity=2)

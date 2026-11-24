"""Discovery coverage is separate from ranking approval and simulated execution."""
import copy
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from . import hotspot_core as H, pipeline
from .fupan import ASSETS, build as B
from .test_fupan import example, frame

DAY = "2026-09-21"
DAYS = [f"2026-09-{d:02d}" for d in (1, 2, 3, 4, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 21)]


class HotspotCoreTests(unittest.TestCase):
    def setUp(self):
        self.f = frame(DAYS, [10.0] * 15, limit_up_days=(DAY,))
        self.facts = {"boards": {"U": ["sh_600001", "sh_688001"], "Z": ["sz_300001"]},
                      "ladder": {"heights": {"sh_600001": 1, "sh_688001": 1}},
                      "themes": {"top_by_limit_up": [{"theme": "robot", "members": 200, "limit_up": 10, "generic": False}],
                                 "top_by_limit_share": []}, "loss_effects": {"events": {}}}
        self.mapping = {s: {"name": s, "concepts": ["robot"]} for s in ("sh_600001", "sh_688001", "sz_300001", "sh_600002")}
        self.rows = {s: {"frame": self.f.copy(), "in_domain": True} for s in self.mapping}

    def build(self, records=None, inputs=None):
        return H.build(DAY, self.rows, self.facts, self.mapping, records or {}, inputs or {})

    def test_first_board_without_new_high_and_growth_boards_survive(self):
        self.rows["sh_600001"]["frame"].loc[0, "high"] = 20
        out = self.build()
        by = {r["symbol"]: r for r in out["rows"]}
        self.assertEqual(set(by), {"sh_600001", "sh_688001", "sz_300001"})
        self.assertIn("首板观察", by["sh_600001"]["roles"][0])
        self.assertIn("触板未封住", by["sz_300001"]["roles"])
        self.assertFalse(out["orders_enabled"])
        self.assertTrue(all(not r["orders_enabled"] for r in by.values()))

    def test_early_s5_survives_outside_all_three_routes_and_hot_themes(self):
        self.mapping["sh_600002"]["concepts"] = []
        out = self.build(inputs={"s5": {"observations": [{"symbol": "sh_600002"}]}})
        row = next(r for r in out["rows"] if r["symbol"] == "sh_600002")
        self.assertEqual(row["engine_state"], "NOT_IN_CANDIDATES")
        self.assertIn("sh_600002", out["early_setups"])
        self.assertEqual(row["themes"], [])

    def test_risk_never_erases_leadership_or_grants_permission(self):
        self.facts["ladder"]["heights"]["sh_600001"] = 3
        r = {"sh_600001": {"state": "RISK_WATCH", "blockers": ["OVERHEATED", "MARKET_WAIT"]}}
        original = copy.deepcopy(r)
        out = self.build(records=r)
        row = next(x for x in out["rows"] if x["symbol"] == "sh_600001")
        self.assertIn("连板核心", row["roles"][0])
        self.assertIn("OVERHEATED", row["risks"])
        self.assertEqual(row["engine_state"], "RISK_WATCH")
        self.assertEqual(r, original)

    def test_unmapped_highest_board_survives_without_any_scenario(self):
        self.mapping["sh_600001"]["concepts"] = ["car_parts"]
        self.facts["ladder"]["heights"]["sh_600001"] = 5
        out = self.build(records={"sh_600001": {"state": "MARKET_WAIT", "blockers": ["MARKET_WAIT"],
                                               "source_record": {"ema20_bias_pct": 43.01}}})
        row = next(x for x in out["rows"] if x["symbol"] == "sh_600001")
        self.assertEqual(row["themes"], [])
        self.assertEqual(row["ema20_bias_pct"], 43.01)
        self.assertEqual(out["focus"][0], "sh_600001")
        self.assertIn("sh_600001", out["market_leaders"])
        self.assertTrue(any(r["symbol"] == "sh_600001" for r in out["mapping_review"]))
        self.assertTrue(any("关联归因待核验" in r for r in row["risks"]))
        self.assertFalse(row["orders_enabled"])
        self.assertEqual(row["engine_state"], "MARKET_WAIT")

    def test_market_leader_survives_when_no_hot_themes_exist(self):
        self.facts["themes"] = {}
        self.facts["ladder"]["heights"]["sh_600001"] = 2
        out = self.build()
        self.assertEqual(out["focus"], ["sh_600001"])
        self.assertEqual([r["symbol"] for r in out["rows"]], ["sh_600001"])

    def test_sourced_narrative_keeps_business_risk_visible(self):
        link = {"themes": ["robot"], "relation_kind": "MARKET_NARRATIVE", "risk_note": "叙事不等于主营"}
        self.mapping["sh_600001"]["_membership_evidence"] = [link]
        row = next(x for x in self.build()["rows"] if x["symbol"] == "sh_600001")
        self.assertEqual(row["theme_evidence"], [link])
        self.assertIn("叙事不等于主营", row["risks"])

    def test_validated_attribution_admits_unmapped_first_board_only_to_observation(self):
        self.mapping["sh_600001"]["concepts"] = []
        attribution = {"known_at": "2026-09-21T21:00:00+08:00", "report_path": "frozen/report",
                       "rows": [{"symbol": "sh_600001", "primary_theme": "new theme", "status": "CORROBORATED"}]}
        original = copy.deepcopy(self.mapping)
        row = next(r for r in self.build(inputs={"attribution": attribution})["rows"] if r["symbol"] == "sh_600001")
        self.assertEqual(row["themes"], ["new theme"])
        self.assertEqual(row["attribution"]["source"], "frozen/report")
        self.assertFalse(row["orders_enabled"])
        self.assertEqual(self.mapping, original)
        attribution["rows"][0]["status"] = "INSUFFICIENT"
        self.assertNotIn("sh_600001", [r["symbol"] for r in self.build(inputs={"attribution": attribution})["rows"]])

    def test_one_price_has_execution_warning_not_false_absorption(self):
        for col in ("open", "high", "low", "close"):
            self.rows["sh_688001"]["frame"].loc[14, col] = 11
        row = next(x for x in self.build()["rows"] if x["symbol"] == "sh_688001")
        self.assertTrue(any("一字价" in x for x in row["risks"]))

    def test_multiple_themes_preserved_generic_excluded(self):
        self.facts["themes"]["top_by_limit_share"] = [{"theme": "humanoid", "members": 50, "limit_up": 4},
                                                       {"theme": "generic", "members": 3000, "limit_up": 40, "generic": True}]
        self.mapping["sh_600001"]["concepts"] = ["generic", "humanoid", "robot"]
        row = next(x for x in self.build()["rows"] if x["symbol"] == "sh_600001")
        self.assertEqual(row["themes"], ["robot", "humanoid"])

    def test_s4_excluded_never_becomes_positive_scenario(self):
        out = self.build(inputs={"s4": {"excluded": [{"symbol": "sh_600002"}], "recommended": []}})
        self.assertNotIn("sh_600002", [r["symbol"] for r in out["rows"]])

    def test_no_stale_bar_or_non_domain_stock(self):
        self.rows["sh_600001"]["frame"] = self.f.iloc[:-1]
        self.rows["sh_688001"]["in_domain"] = False
        out = self.build()
        self.assertEqual(out["unavailable"], ["sh_600001", "sh_688001"])
        self.assertEqual([r["symbol"] for r in out["rows"]], ["sz_300001"])

    def test_duplicate_evidence_is_one_record_with_multiple_labels(self):
        out = self.build(inputs={"strong": {"boards": {"attention": [{"symbol": "sh_600001", "entry": "BREAKOUT"}]}},
                                 "s1": {"observations": [{"symbol": "sh_600001"}]}})
        self.assertEqual(len(out["rows"]), 3)
        r = next(r for r in out["rows"] if r["symbol"] == "sh_600001")
        self.assertIn("S1", r["roles"])
        self.assertIn("趋势突破", r["roles"])
        self.assertEqual(len(out["focus"]), len(set(out["focus"])))

    def test_load_rejects_wrong_dates_and_failed_rerun_same_day(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "strong_trend_shadow.json").write_text(json.dumps({"as_of": DAY}), encoding="utf-8")
            (p / "s1_observations.json").write_text(json.dumps({"as_of": DAY}), encoding="utf-8")
            (p / "s5_observations.json").write_text(json.dumps({"as_of": "2026-09-18"}), encoding="utf-8")
            (p / "mainwave_run_status.json").write_text(json.dumps({"as_of": DAY, "steps": [{"name": "strong_trend_shadow", "status": "WARNING"}]}), encoding="utf-8")
            inputs, gaps = H.load_inputs(p, DAY)
            self.assertNotIn("strong", inputs)
            self.assertNotIn("s1", inputs)
            self.assertNotIn("s5", inputs)
            self.assertTrue(gaps)

    def test_attribution_warning_accepts_partial_data_but_rejects_failed_run_leftovers(self):
        from . import eod_review as R
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            R._write(folder / "theme_attribution.json", {"as_of": DAY, "status": "PARTIAL", "report_path": "v001"})
            R._write(folder / "mainwave_run_status.json", {"as_of": DAY, "steps": [{"name": "limit_theme", "status": "WARNING"}]})
            health = {"as_of": DAY, "status": "PARTIAL", "path": "v001", "scope": {"full_limit_up_coverage": True}}
            R._write(folder / "limit_theme/status.json", health)
            self.assertIn("attribution", H.load_inputs(folder, DAY)[0])
            for status, path, full in (("FAILED", "v001", True), ("RUNNING", "v001", True), ("PARTIAL", "v002", True), ("PARTIAL", "v001", False)):
                R._write(folder / "limit_theme/status.json", {**health, "status": status, "path": path, "scope": {"full_limit_up_coverage": full}})
                with self.subTest(status=status, path=path, full=full):
                    self.assertNotIn("attribution", H.load_inputs(folder, DAY)[0])

    def test_card_leader_not_mislabelled_pullback_without_plan(self):
        card = B.stock_card("sh_600001", "test", self.rows["sh_600001"], {"mode": "TREND_PULLBACK", "state": "RISK_WATCH"},
                            None, None, None, None, {}, DAY, "2026-09-22", True, B.Sources(), [],
                            opp={"score": 60, "text": "test", "parts": [("3 板", 15)]})
        self.assertEqual(card["mode"], "STRONG_CONFIRM")
        self.assertIsNone(card["plan"])

    def test_outer_s4_failure_rejects_same_day_leftover(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "s4_opportunities.json").write_text(json.dumps({"as_of": DAY, "recommended": [{"symbol": "sh_600001"}]}), encoding="utf-8")
            (p / "daily_full_status.json").write_text(json.dumps({"as_of": DAY, "steps": [{"name": "s4_opportunities", "status": "FAILED"}]}), encoding="utf-8")
            inputs, gaps = H.load_inputs(p, DAY)
            self.assertNotIn("s4", inputs)
            self.assertTrue(any("s4_opportunities" in g for g in gaps))

    def test_observation_failure_warns_but_does_not_stop_eod(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(pipeline, "OUT_ROOT", Path(tmp)):
            def launch(command, check=False):
                return SimpleNamespace(returncode=1 if command[2] == "mainwave_engine.strong_trend" else 0)
            with patch.object(pipeline.subprocess, "run", side_effect=launch) as run:
                self.assertEqual(pipeline.run(DAY), 0)
            mods = [call.args[0][2] for call in run.call_args_list]
            self.assertLess(mods.index("mainwave_engine.scenario_scan"), mods.index("mainwave_engine.fupan"))
            state = json.loads((Path(tmp) / "20260921/mainwave_run_status.json").read_text(encoding="utf-8"))
            self.assertTrue(any("strong_trend_shadow" in x for x in state["warnings"]))

    def test_sidecar_survives_ai_interpretation_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = {"input_hash": "new", "inputs": {"x": 1}, "live": False, "hotspot_core": self.build(),
                    "entry_review": {"eligible_count": 0, "top": [], "orders_enabled": False}}
            first = B.persist(DAY, example(), meta, tmp)
            second = B.apply_analysis(DAY, {}, tmp)
            for result in (first, second):
                core = json.loads((Path(result["path"]) / "hotspot_core.json").read_text(encoding="utf-8"))
                self.assertEqual(core, meta["hotspot_core"])
                entry = json.loads((Path(result["path"]) / "entry_review.json").read_text(encoding="utf-8"))
                self.assertEqual(entry, meta["entry_review"])

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for the DOM-free filter test")
    def test_focus_filter_searches_full_pool_and_can_show_all(self):
        # Unit-test the event handlers with a tiny document double: no browser, filesystem page or network.
        script = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const handlers={},nodes={};
function node(id){return nodes[id]||(nodes[id]={textContent:'{}',addEventListener(){}})}
const rows=[{textContent:'赛福天 robot',dataset:{hotspotFocus:'1'},classList:{toggle(){}}},
            {textContent:'光洋股份 sz_002708 robot',dataset:{hotspotFocus:'0'},classList:{toggle(){}}}];
const context={document:{querySelector:s=>node(s),querySelectorAll:s=>s==='[data-hotspot-row]'?rows:[],
                        addEventListener:(name,fn)=>handlers[name]=fn},
               window:{addEventListener(){}},console};
vm.createContext(context);vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context);
handlers.input({target:{id:'hotspotSearch',value:''}});
assert.deepEqual(rows.map(r=>r.hidden),[false,true]);
handlers.input({target:{id:'hotspotSearch',value:'光洋'}});
assert.deepEqual(rows.map(r=>r.hidden),[true,false]);
assert.ok(nodes['#hotspotCount'].textContent.includes('全池搜索'));
handlers.input({target:{id:'hotspotSearch',value:''}});
handlers.change({target:{id:'hotspotFocus',checked:false}});
assert.deepEqual(rows.map(r=>r.hidden),[false,false]);
"""
        run = subprocess.run([shutil.which("node"), "-e", script, str(ASSETS / "report.js")], capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(run.returncode, 0, run.stderr)


if __name__ == "__main__":
    unittest.main()

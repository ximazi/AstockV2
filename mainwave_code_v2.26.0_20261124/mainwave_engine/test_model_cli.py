"""Local Codex transport and manual evidence assessment without network calls."""
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from . import model_cli as C
from .trading import assessment as A

CFG = {"provider": "codex-cli", "model": C.DEFAULT_MODEL, "timeout": 30, "proxy": "http://127.0.0.1:7890"}
AT = "2026-10-29T01:00:00+08:00"


class TransportTest(unittest.TestCase):
    @staticmethod
    def done(command, answer='{"ok":true}', events=None, code=0, **kwargs):
        if answer is not None:
            Path(command[command.index("--output-last-message") + 1]).write_text(answer, encoding="utf-8")
        if events is None:
            events = [{"type": "item.completed", "item": {"type": "agent_message", "text": answer}},
                      {"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 4}}]
        return SimpleNamespace(returncode=code, stdout="\n".join(json.dumps(e) for e in events), stderr="secret-stderr")

    def test_config_uses_codex_instead_of_keys_or_old_providers(self):
        for provider in (None, "auto", "claude-cli", "openai", "anthropic", "codex-cli"):
            env = {"OPENAI_API_KEY": "fake", "ANTHROPIC_API_KEY": "fake"}
            if provider:
                env["FUPAN_ANALYZER"] = provider
            self.assertEqual(C.config(env)["provider"], "codex-cli")
        self.assertEqual(C.config({})["model"], C.DEFAULT_MODEL)
        self.assertEqual(C.config({"FUPAN_ANALYZER": "off"})["provider"], "off")
        self.assertEqual(C.config({"FUPAN_MODEL": "explicit"})["model"], "explicit")

    def test_native_executable_resolution_avoids_npm_shell(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            binary = root / "node_modules/@openai/codex/node_modules/@openai/codex-win32-x64/vendor/x64/bin/codex.exe"
            binary.parent.mkdir(parents=True)
            binary.touch()
            with patch.object(C.shutil, "which", return_value=str(root / "codex.cmd")):
                self.assertEqual(C.executable(), str(binary))
        with patch.object(C.shutil, "which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "not found"):
                C.executable()

    def test_stdin_isolated_readonly_command_and_child_only_proxy(self):
        before = dict(os.environ)
        with patch.object(C, "executable", return_value="codex.exe"), \
                patch.object(C.subprocess, "run", side_effect=self.done) as run:
            text, usage = C.call("system", "untrusted evidence", CFG)
        command, kwargs = run.call_args.args[0], run.call_args.kwargs
        self.assertEqual(json.loads(text), {"ok": True})
        self.assertEqual(usage["input_tokens"], 10)
        self.assertIn("--ignore-user-config", command)
        self.assertIn("--ephemeral", command)
        self.assertEqual(command[command.index("--sandbox") + 1], "read-only")
        self.assertEqual(command[command.index("-a") + 1], "never")
        self.assertIn('web_search="disabled"', command)
        self.assertIn("shell_tool", command)
        self.assertIn("hooks", command)
        self.assertNotIn("--model", command)
        self.assertEqual(command[-1], "-")
        self.assertNotIn("untrusted evidence", " ".join(command))
        self.assertIn("untrusted evidence", kwargs["input"])
        self.assertEqual(kwargs["env"]["HTTPS_PROXY"], CFG["proxy"])
        self.assertNotIn("OPENAI_API_KEY", kwargs["env"])
        self.assertNotEqual(Path(kwargs["cwd"]), Path.cwd())
        self.assertFalse(Path(kwargs["cwd"]).exists())
        self.assertEqual(dict(os.environ), before)

    def test_explicit_model(self):
        with patch.object(C, "executable", return_value="codex.exe"), \
                patch.object(C.subprocess, "run", side_effect=self.done) as run:
            C.call("s", "u", {**CFG, "model": "explicit"})
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--model") + 1], "explicit")

    def test_incomplete_failed_tool_or_empty_answer_is_rejected(self):
        cases = [(None, [], 0), ("", None, 0), ("{}", [{"type": "turn.failed"}], 0),
                 ("{}", [{"type": "error"}], 0), ("{}", None, 1),
                 ("{}", [{"type": "item.started", "item": {"type": "command_execution"}},
                         {"type": "turn.completed"}], 0)]
        for answer, events, code in cases:
            with self.subTest(answer=answer, events=events, code=code), \
                    patch.object(C, "executable", return_value="codex.exe"), \
                    patch.object(C.subprocess, "run", side_effect=lambda cmd, **kw: self.done(
                        cmd, answer=answer, events=events, code=code, **kw)):
                with self.assertRaises(RuntimeError) as raised:
                    C.call("s", "u", CFG)
                self.assertNotIn("secret-stderr", str(raised.exception))

    def test_timeout_is_explicit(self):
        with patch.object(C, "executable", return_value="codex.exe"), \
                patch.object(C.subprocess, "run", side_effect=subprocess.TimeoutExpired("codex", 1)):
            with self.assertRaises(TimeoutError):
                C.call("s", "u", CFG)


class ConfiguredAssessmentTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / "audit.sqlite3"
        self.pack = {"evidence_hash": "h", "opportunity_id": "o", "as_of": "2026-10-28",
                     "evidence": [{"id": "E01", "as_of": "2026-10-28"}]}

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def answer(system, user, cfg):
        envelope = json.loads(user)
        return json.dumps({"verdict": "WAIT", "supporting_evidence": [{"evidence_id": "E01", "reason": "fact"}],
                           "counter_evidence": [], "unknowns": ["confirmation"], "alternative_explanations": [],
                           "mode_fit": "UNKNOWN", "next_review_event": "next close", "expires_at": envelope["expires_at"]}), {}

    def test_configured_codex_answer_validates_and_caches(self):
        with patch.object(C, "call", side_effect=self.answer) as call:
            first = A.configured_assess(self.pack, self.db, AT, CFG)
            again = A.configured_assess(self.pack, self.db, AT, CFG)
        self.assertEqual(call.call_count, 1)
        self.assertEqual(first["assessment"]["verdict"], "WAIT")
        self.assertTrue(first["validation"]["valid"])
        self.assertTrue(again["cached"])
        self.assertTrue(first["model"].startswith("codex-cli:"))
        self.assertIsNone(first["cost"]["amount"])
        self.assertFalse(A.gate(first, required=True)["new_entry_allowed"])

    def test_failed_call_is_unknown_and_retry_keeps_failure_history(self):
        with patch.object(C, "call", side_effect=TimeoutError):
            first = A.configured_assess(self.pack, self.db, AT, CFG)
        self.assertEqual((first["status"], first["assessment"]["verdict"]), ("TIMEOUT", "UNKNOWN"))
        with patch.object(C, "call", side_effect=self.answer) as call:
            second = A.configured_assess(self.pack, self.db, AT, CFG)
            third = A.configured_assess(self.pack, self.db, AT, CFG)
        self.assertEqual(call.call_count, 1)
        self.assertEqual(second["status"], "OK")
        self.assertTrue(third["cached"])
        self.assertEqual(len(A.history("o", self.db)["assessments"]), 2)

    def test_invalid_citations_or_expiry_do_not_support_trades(self):
        for replacement in ({"supporting_evidence": [{"evidence_id": "invented"}], "verdict": "SUPPORT"},
                            {"expires_at": "bad-time"}, {"supporting_evidence": [{"evidence_id": []}]}):
            def bad(s, u, c):
                text, usage = self.answer(s, u, c)
                return json.dumps({**json.loads(text), **replacement}), usage
            with self.subTest(replacement=replacement), patch.object(C, "call", side_effect=bad):
                result = A.configured_assess(self.pack, self.db, AT, CFG)
                self.assertEqual(result["assessment"]["verdict"], "UNKNOWN")

    def test_off_never_calls_cli(self):
        with patch.object(C, "call") as call:
            result = A.configured_assess(self.pack, self.db, AT, {"provider": "off"})
        call.assert_not_called()
        self.assertEqual(result["assessment"]["verdict"], "UNKNOWN")


if __name__ == "__main__":
    unittest.main()

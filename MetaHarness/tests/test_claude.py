"""P24 Claude Code runtime and reviser tests; no real Claude process is used."""

from __future__ import annotations

import json
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness.claude.agent import (  # noqa: E402
    ClaudeCodeAgent,
    build_claude_environment,
    build_revision_prompt,
    parse_scope_request,
)
from metaharness.claude.auth import check_claude_authentication  # noqa: E402
from metaharness.claude.runtime import (  # noqa: E402
    ClaudeRuntimeError,
    prepare_claude_home,
)
from metaharness.config import load_config  # noqa: E402
from metaharness.models import (  # noqa: E402
    ClaudeRuntimeConfig,
    ContextConfig,
    HarnessConfig,
    ModelProfile,
    ProfileDriver,
    ExecutionRole,
    SelectionMode,
    CodexRuntimeConfig,
)


class ClaudeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        subprocess.run(["git", "-C", str(self.repo), "config", "user.email", "test@example.invalid"], check=True)
        subprocess.run(["git", "-C", str(self.repo), "config", "user.name", "test"], check=True)
        (self.repo / "README.md").write_text("base\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(self.repo), "add", "README.md"], check=True)
        subprocess.run(["git", "-C", str(self.repo), "commit", "-qm", "base"], check=True)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _config(self) -> HarnessConfig:
        return HarnessConfig(
            repo=self.repo,
            base_ref="HEAD",
            runs_root=self.root / "runs",
            worktrees_root=self.root / "worktrees",
            require_clean_base=True,
            context=ContextConfig(always_files=()),
            check_catalog=(),
            allow_no_required_checks=True,
            codex_runtime=CodexRuntimeConfig(self.root / "codex-home"),
            claude_runtime=ClaudeRuntimeConfig(self.root / "claude-home"),
        )

    def _executable(self, name: str, body: str) -> Path:
        path = self.root / name
        path.write_text("#!/usr/bin/python3.12\n" + textwrap.dedent(body), encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        return path

    def test_managed_home_and_environment_are_isolated(self) -> None:
        home = prepare_claude_home(self._config())
        self.assertEqual(home, (self.root / "claude-home").resolve())
        self.assertEqual(
            (home / "settings.json").read_text(),
            '{\n'
            '  "$schema": "https://json.schemastore.org/claude-code-settings.json",\n'
            '  "permissions": {\n'
            '    "disableBypassPermissionsMode": "disable",\n'
            '    "deny": [\n'
            '      "Agent",\n'
            '      "AskUserQuestion",\n'
            '      "Bash",\n'
            '      "WebFetch",\n'
            '      "WebSearch"\n'
            '    ]\n'
            '  }\n'
            '}\n',
        )
        settings = json.loads((home / "settings.json").read_text())
        self.assertEqual(settings["permissions"]["disableBypassPermissionsMode"], "disable")
        self.assertNotIn("disableBypassPermissionsMode", settings)
        self.assertEqual((home / "empty-mcp.json").read_text(), '{\n  "mcpServers": {}\n}\n')
        environment = build_claude_environment(
            {
                "PATH": "/bin",
                "HOME": "/personal",
                "LANG": "C",
                "CODEX_HOME": "/personal/codex",
                "BRIDGE_API_KEY": "secret",
                "ANTHROPIC_API_KEY": "secret",
                "OPENAI_API_KEY": "secret",
                "PERSONAL_SETTING": "no",
            },
            claude_home=home,
        )
        # P28: HOME, cache and TMPDIR are forced below the managed home; the
        # personal HOME is never inherited.
        self.assertEqual(
            environment,
            {
                "PATH": "/bin",
                "LANG": "C",
                "HOME": str(home / "home"),
                "CLAUDE_CONFIG_DIR": str(home),
                "XDG_CACHE_HOME": str(home / "cache"),
                "TMPDIR": str(home / "tmp"),
            },
        )

    def test_managed_settings_reject_symlinks_and_replace_divergent_bytes(self) -> None:
        home = self.root / "claude-home"
        home.mkdir()
        target = self.root / "personal-settings.json"
        target.write_text('{"personal": true}\n', encoding="utf-8")
        (home / "settings.json").symlink_to(target)
        with self.assertRaisesRegex(ClaudeRuntimeError, "settings.json must not be a symlink"):
            prepare_claude_home(self._config())

        (home / "settings.json").unlink()
        (home / "settings.json").mkdir()
        with self.assertRaisesRegex(ClaudeRuntimeError, "settings.json must be a regular file"):
            prepare_claude_home(self._config())
        (home / "settings.json").rmdir()
        (home / "settings.json").write_text('{"permissions": {"deny": []}}\n', encoding="utf-8")
        credentials = home / "credentials.json"
        credentials.write_text("credential-state\n", encoding="utf-8")
        prepared = prepare_claude_home(self._config())
        self.assertEqual(prepared / "settings.json", home.resolve() / "settings.json")
        self.assertIn('"disableBypassPermissionsMode": "disable"', (home / "settings.json").read_text())
        self.assertEqual(credentials.read_text(), "credential-state\n")

    def test_exact_argv_stdin_stream_result_and_usage(self) -> None:
        capture = self.root / "capture.json"
        executable = self._executable(
            "claude",
            f"""
            import json, pathlib, sys
            pathlib.Path({str(capture)!r}).write_text(json.dumps({{"argv": sys.argv[1:], "stdin": sys.stdin.read()}}))
            print(json.dumps({{"type": "unknown.event", "opaque": "ok"}}))
            print(json.dumps({{"type": "result", "result": "revised", "usage": {{"input_tokens": 4, "output_tokens": 6, "cache_read_input_tokens": 2}}}}))
            """,
        )
        profile = ModelProfile(
            id="claude",
            display_name="Claude",
            roles=(ExecutionRole.REVISER,),
            driver=ProfileDriver.CLAUDE_CODE,
            model="opus",
            selection_mode=SelectionMode.CLI,
            effort="medium",
            permission_mode="acceptEdits",
            timeout_seconds=5,
            retries=0,
        )
        home = prepare_claude_home(self._config())
        result = ClaudeCodeAgent(executable=str(executable)).run_revision(
            "inspect this", self.repo, artifacts_dir=self.root / "run", profile=profile,
            environment=build_claude_environment({"PATH": "/usr/bin"}, claude_home=home),
        )
        recorded = json.loads(capture.read_text())
        self.assertEqual(recorded["stdin"].splitlines()[0], "inspect this")
        self.assertEqual(
            recorded["argv"],
            ["--print", "--verbose", "--output-format", "stream-json", "--safe-mode", "--restricted", "--tools", "Read,Edit,Write,Grep,Glob",
             "--no-session-persistence", "--no-chrome", "--disable-slash-commands",
             "--model", "opus", "--effort", "medium",
             "--permission-mode", "acceptEdits", "--settings", str(home / "settings.json"),
             "--strict-mcp-config", "--mcp-config", str(home / "empty-mcp.json")],
        )
        self.assertNotIn("--max-turns", recorded["argv"])
        for option in ("--model", "--effort", "--permission-mode", "--restricted", "--safe-mode"):
            self.assertIn(option, recorded["argv"])
        self.assertIn("--tools", recorded["argv"])
        self.assertEqual(
            recorded["argv"][recorded["argv"].index("--tools") + 1],
            "Read,Edit,Write,Grep,Glob",
        )
        self.assertEqual(result.final_message, "revised")
        self.assertEqual(result.usage["cached_input_tokens"], 2)
        self.assertTrue((self.root / "run/revision/agent.events.jsonl").exists())

    def test_terminal_error_max_turns_is_persisted(self) -> None:
        executable = self._executable(
            "claude-terminal-error",
            """
            import json
            print(json.dumps({
                "type": "result",
                "subtype": "error_max_turns",
                "is_error": True,
                "num_turns": 13,
                "stop_reason": "max_turns",
                "errors": ["turn limit reached", 17],
                "unknown": "not persisted",
            }))
            """,
        )
        profile = ModelProfile(
            id="claude",
            display_name="Claude",
            roles=(ExecutionRole.REVISER,),
            driver=ProfileDriver.CLAUDE_CODE,
            model="opus",
            selection_mode=SelectionMode.CLI,
            effort="medium",
            permission_mode="acceptEdits",
            timeout_seconds=5,
            retries=0,
        )
        home = prepare_claude_home(self._config())

        result = ClaudeCodeAgent(executable=str(executable)).run_revision(
            "inspect this", self.repo, artifacts_dir=self.root / "terminal-error-run",
            profile=profile,
            environment=build_claude_environment({"PATH": "/usr/bin"}, claude_home=home),
        )

        persisted = json.loads(
            (self.root / "terminal-error-run/revision/agent.result.json").read_text()
        )
        self.assertEqual(result.terminal_type, "result")
        self.assertEqual(result.terminal_subtype, "error_max_turns")
        self.assertTrue(result.terminal_is_error)
        self.assertEqual(result.terminal_num_turns, 13)
        self.assertEqual(result.terminal_stop_reason, "max_turns")
        self.assertEqual(result.terminal_errors, ("turn limit reached",))
        self.assertEqual(persisted["terminal_subtype"], "error_max_turns")
        self.assertEqual(persisted["terminal_num_turns"], 13)
        self.assertEqual(persisted["terminal_errors"], ["turn limit reached"])
        self.assertNotIn("unknown", persisted)

    def test_profile_timeout_is_passed_to_run_bounded(self) -> None:
        profile = ModelProfile(
            id="claude",
            display_name="Claude",
            roles=(ExecutionRole.REVISER,),
            driver=ProfileDriver.CLAUDE_CODE,
            model="opus",
            selection_mode=SelectionMode.CLI,
            effort="medium",
            permission_mode="acceptEdits",
            timeout_seconds=37,
            retries=0,
        )
        home = prepare_claude_home(self._config())
        environment = build_claude_environment({"PATH": "/usr/bin"}, claude_home=home)

        with patch("metaharness.claude.agent.run_bounded", return_value=(9, False)) as bounded:
            result = ClaudeCodeAgent().run_revision(
                "inspect this", self.repo, artifacts_dir=self.root / "timeout-run",
                profile=profile, environment=environment,
            )

        self.assertEqual(result.exit_code, 9)
        self.assertEqual(bounded.call_args.kwargs["timeout_seconds"], 37)

    def test_subscription_auth_uses_safe_mode_without_copying_credentials(self) -> None:
        capture = self.root / "subscription-capture.json"
        executable = self._executable(
            "claude",
            f"""
            import json, os, pathlib, sys
            args = sys.argv[1:]
            if args == ["auth", "--help"]:
                print("Commands: status")
                raise SystemExit(0)
            if args == ["auth", "status"]:
                print("Logged in")
                raise SystemExit(0)
            pathlib.Path({str(capture)!r}).write_text(json.dumps({{
                "argv": args,
                "env": dict(os.environ),
            }}))
            print(json.dumps({{"type": "result", "result": "ok"}}))
            """,
        )
        home = prepare_claude_home(self._config())
        managed_credentials = home / ".credentials.json"
        managed_credentials.write_text("managed-subscription-credential\n", encoding="utf-8")
        credentials_before = managed_credentials.read_bytes()
        source = {
            "PATH": str(self.root),
            "HOME": str(self.root / "personal"),
            "ANTHROPIC_API_KEY": "must-not-propagate",
            "OPENAI_API_KEY": "must-not-propagate",
        }
        environment = build_claude_environment(source, claude_home=home)

        auth_status = check_claude_authentication(home, environment=environment)
        self.assertTrue(auth_status.available)
        profile = ModelProfile(
            id="claude",
            display_name="Claude",
            roles=(ExecutionRole.REVISER,),
            driver=ProfileDriver.CLAUDE_CODE,
            model="opus",
            selection_mode=SelectionMode.CLI,
            effort="medium",
            permission_mode="acceptEdits",
            timeout_seconds=5,
            retries=0,
        )
        ClaudeCodeAgent(executable=str(executable)).run_revision(
            "inspect this", self.repo, artifacts_dir=self.root / "subscription-run",
            profile=profile, environment=environment,
        )

        recorded = json.loads(capture.read_text())
        self.assertIn("--safe-mode", recorded["argv"])
        self.assertNotIn("--bare", recorded["argv"])
        self.assertIn("--restricted", recorded["argv"])
        self.assertEqual(recorded["env"]["CLAUDE_CONFIG_DIR"], str(home))
        self.assertNotIn("ANTHROPIC_API_KEY", recorded["env"])
        self.assertNotIn("OPENAI_API_KEY", recorded["env"])
        self.assertEqual(managed_credentials.read_bytes(), credentials_before)

    def test_revision_prompt_keeps_semantics_without_repeating_removed_tools(self) -> None:
        rendered = build_revision_prompt("review request")
        self.assertIn("authorized by the implementation contract", rendered)
        self.assertIn("Do not create commits", rendered)
        self.assertIn("deterministic checks", rendered)
        self.assertIn("Runtime capabilities are deliberately restricted", rendered)
        self.assertNotIn("Do not execute Bash", rendered)

    def test_claude_profile_config_rejects_forbidden_fields(self) -> None:
        config = self.root / "bad.toml"
        config.write_text(
            f"""
            repo = {str(self.repo)!r}
            base_ref = "HEAD"
            runs_root = {str(self.root / 'runs')!r}
            worktrees_root = {str(self.root / 'worktrees')!r}
            allow_no_required_checks = true
            [ui]
            default_planner_profile = "p"
            default_implementer_profile = "i"
            default_reviewer_profile = "p"
            [model_profiles.p]
            display_name = "p"
            roles = ["planner", "reviewer"]
            driver = "openai-chat"
            model = "p"
            selection_mode = "request"
            base_url = "https://p.invalid"
            endpoint_path = "/v1"
            [model_profiles.i]
            display_name = "i"
            roles = ["implementer"]
            driver = "codex"
            model = "i"
            effort = "high"
            sandbox = "workspace-write"
            selection_mode = "cli"
            [model_profiles.c]
            display_name = "c"
            roles = ["reviser"]
            driver = "claude-code"
            model = "opus"
            effort = "medium"
            permission_mode = "acceptEdits"
            selection_mode = "cli"
            sandbox = "read-only"
            """,
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "sandbox is not allowed"):
            load_config(config)


class ScopeRequestParserTests(unittest.TestCase):
    """``META SCOPE REQUEST v1`` syntax, tested away from the pipeline.

    The parser is a Claude primitive: a malformed block must be
    indistinguishable from no block at all, and no input may raise.
    """

    VALID = """META SCOPE REQUEST v1

REASON
The failing correction requires the existing test.

PATHS
- frontend/e2e/example.spec.ts

EVIDENCE
- CHECK_FAILED:test | locator no longer matches

END META SCOPE REQUEST
"""

    def parse(self, text: str):
        try:
            return parse_scope_request(text)
        except Exception as exc:  # pragma: no cover - the assertion is the test
            self.fail(f"parse_scope_request raised {exc!r}")

    def test_a_valid_block_yields_the_exact_reason_paths_and_evidence(self) -> None:
        request = self.parse(self.VALID)

        self.assertIsNotNone(request)
        self.assertEqual(request.reason, "The failing correction requires the existing test.")
        self.assertEqual(request.paths, ("frontend/e2e/example.spec.ts",))
        self.assertEqual(
            request.evidence, ("CHECK_FAILED:test | locator no longer matches",)
        )

    def test_a_valid_block_survives_surrounding_prose(self) -> None:
        request = self.parse("Report intro.\n\n" + self.VALID + "\nTrailing prose.\n")

        self.assertIsNotNone(request)
        self.assertEqual(request.paths, ("frontend/e2e/example.spec.ts",))

    def replace_paths(self, *paths: str) -> str:
        return self.VALID.replace(
            "- frontend/e2e/example.spec.ts",
            "\n".join(f"- {path}" for path in paths),
        )

    def test_every_rejected_path_shape_is_indistinguishable_from_no_request(self) -> None:
        cases = {
            "duplicate": self.replace_paths("src/a.py", "src/a.py"),
            "absolute": self.replace_paths("/etc/passwd"),
            "traversal": self.replace_paths("../outside.py"),
            "dot slash": self.replace_paths("./foo.py"),
            "directory": self.replace_paths("src/"),
            "backslash": self.replace_paths("src\\a.py"),
            "glob star": self.replace_paths("src/*.py"),
            "glob question": self.replace_paths("src/a?.py"),
            "glob bracket": self.replace_paths("src/[ab].py"),
            "glob brace": self.replace_paths("src/{a,b}.py"),
            "empty path entry": self.replace_paths(" "),
            "more than 32 paths": self.replace_paths(
                *[f"src/module_{index}.py" for index in range(33)]
            ),
        }
        for name, text in cases.items():
            with self.subTest(name):
                self.assertIsNone(self.parse(text), name)

    def test_exactly_32_paths_are_still_accepted(self) -> None:
        request = self.parse(
            self.replace_paths(*[f"src/module_{index}.py" for index in range(32)])
        )

        self.assertIsNotNone(request)
        self.assertEqual(len(request.paths), 32)

    def test_every_malformed_envelope_is_indistinguishable_from_no_request(self) -> None:
        cases = {
            "duplicate header": self.VALID.replace(
                "META SCOPE REQUEST v1",
                "META SCOPE REQUEST v1\n\nMETA SCOPE REQUEST v1",
                1,
            ),
            "missing footer": self.VALID.replace("END META SCOPE REQUEST\n", ""),
            "duplicate footer": self.VALID + "END META SCOPE REQUEST\n",
            "missing REASON": self.VALID.replace("REASON\n", ""),
            "missing PATHS": self.VALID.replace(
                "PATHS\n- frontend/e2e/example.spec.ts\n\n", ""
            ),
            "missing EVIDENCE": self.VALID.replace(
                "EVIDENCE\n- CHECK_FAILED:test | locator no longer matches\n\n", ""
            ),
            "blank evidence": self.VALID.replace(
                "- CHECK_FAILED:test | locator no longer matches", "- "
            ),
            "blank reason": self.VALID.replace(
                "The failing correction requires the existing test.", "   "
            ),
            "prose inside PATHS": self.VALID.replace(
                "- frontend/e2e/example.spec.ts",
                "the spec file below is required\n- frontend/e2e/example.spec.ts",
            ),
            "prose inside EVIDENCE": self.VALID.replace(
                "- CHECK_FAILED:test | locator no longer matches",
                "the locator drifted\n- CHECK_FAILED:test | locator no longer matches",
            ),
            "footer before header": (
                "END META SCOPE REQUEST\n" + self.VALID.replace(
                    "END META SCOPE REQUEST\n", ""
                )
            ),
            "no block at all": "Plain revision report with no structured block.\n",
            "empty": "",
        }
        for name, text in cases.items():
            with self.subTest(name):
                self.assertIsNone(self.parse(text), name)

    def test_a_non_string_input_returns_none_without_raising(self) -> None:
        for value in (None, 17, [], {"reason": "x"}, object()):
            with self.subTest(repr(value)):
                self.assertIsNone(self.parse(value))


if __name__ == "__main__":
    unittest.main()

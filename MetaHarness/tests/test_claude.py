"""P24 Claude Code runtime and reviser tests; no real Claude process is used."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness.claude.agent import (  # noqa: E402
    ClaudeCodeAgent,
    build_claude_environment,
    build_revision_prompt,
)
from metaharness.claude.auth import check_claude_authentication  # noqa: E402
from metaharness.claude.runtime import (  # noqa: E402
    ClaudeRuntimeError,
    prepare_claude_home,
)
from metaharness.config import load_config  # noqa: E402
from metaharness.models import (  # noqa: E402
    AgentConfig,
    ClaudeRuntimeConfig,
    ContextConfig,
    HarnessConfig,
    LLMEndpointConfig,
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
            planner=LLMEndpointConfig("https://planner.invalid", "/v1", "planner"),
            reviewer=LLMEndpointConfig("https://reviewer.invalid", "/v1", "reviewer"),
            context=ContextConfig(always_files=()),
            agent=AgentConfig(),
            checks=(),
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
             "--no-session-persistence", "--no-chrome", "--disable-slash-commands", "--max-turns", "12",
             "--model", "opus", "--effort", "medium",
             "--permission-mode", "acceptEdits", "--settings", str(home / "settings.json"),
             "--strict-mcp-config", "--mcp-config", str(home / "empty-mcp.json")],
        )
        self.assertEqual(result.final_message, "revised")
        self.assertEqual(result.usage["cached_input_tokens"], 2)
        self.assertTrue((self.root / "run/revision/agent.events.jsonl").exists())

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
            [planner]
            base_url = "https://planner.invalid"
            endpoint_path = "/v1"
            model = "planner"
            [reviewer]
            base_url = "https://reviewer.invalid"
            endpoint_path = "/v1"
            model = "reviewer"
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


if __name__ == "__main__":
    unittest.main()

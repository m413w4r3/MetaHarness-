import json
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness.agent.base import AgentResult
from metaharness.agent.codex import AgentCommittedError, CodexAgent
from metaharness.models import AgentConfig


def run_git(directory: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(directory), *args],
        check=True,
        capture_output=True,
        text=True,
    )


class CodexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        run_git(self.root, "init", "-q")
        run_git(self.root, "config", "user.email", "test@example.invalid")
        run_git(self.root, "config", "user.name", "MetaHarness test")
        (self.root / "README.md").write_text("base\n", encoding="utf-8")
        run_git(self.root, "add", "README.md")
        run_git(self.root, "commit", "-qm", "base")
        self.base_sha = run_git(self.root, "rev-parse", "HEAD").stdout.strip()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def executable(self, body: str) -> Path:
        path = self.root / "fake-codex"
        path.write_text("#!/usr/bin/env python3\n" + textwrap.dedent(body), encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        return path

    def agent(self, executable: Path, timeout: int = 10, **kwargs: object) -> CodexAgent:
        return CodexAgent(
            AgentConfig(timeout_seconds=timeout),
            executable=str(executable),
            **kwargs,
        )

    def test_stdin_args_events_usage_and_final_artifacts(self) -> None:
        prompt_capture = self.root / "prompt"
        args_capture = self.root / "args"
        executable = self.executable(
            f"""
            import json, pathlib, sys
            pathlib.Path({str(prompt_capture)!r}).write_text(sys.stdin.read())
            pathlib.Path({str(args_capture)!r}).write_text(json.dumps(sys.argv[1:]))
            final = pathlib.Path(sys.argv[sys.argv.index('--output-last-message') + 1])
            final.write_text('final from codex\\n')
            print('not an event', flush=True)
            print(json.dumps({{'type': 'turn.completed', 'usage': {{'input_tokens': 3, 'output_tokens': 4, 'total_tokens': 7}}}}), flush=True)
            """
        )
        artifacts = self.root / "artifacts"

        result = self.agent(executable).run("STATUS: READY", self.root, artifacts)

        self.assertIsInstance(result, AgentResult)
        self.assertEqual(result.exit_code, 0)
        self.assertFalse(result.timed_out)
        self.assertEqual(result.final_message, "final from codex\n")
        self.assertEqual(result.usage["total_tokens"], 7)
        prompt = (artifacts / "agent.prompt.txt").read_text()
        self.assertIn("<AUTHORITATIVE IMPLEMENTATION CONTRACT>\nSTATUS: READY", prompt)
        args = json.loads(args_capture.read_text())
        self.assertIn("--json", args)
        self.assertIn("--sandbox", args)
        self.assertIn("workspace-write", args)
        self.assertIn("-m", args)
        self.assertIn("gpt-5.6-luna", args)
        self.assertIn('model_reasoning_effort="high"', args)
        self.assertIn("-C", args)
        self.assertIn(str(self.root), args)
        self.assertEqual(args[-1], "-")
        self.assertEqual((artifacts / "agent.events.jsonl").read_text().splitlines()[0], "not an event")
        saved = json.loads((artifacts / "agent.result.json").read_text())
        self.assertEqual(saved["exit_code"], 0)

    def test_exit_one_is_reported(self) -> None:
        executable = self.executable(
            """
            import sys
            sys.stderr.write('failed\\n')
            sys.exit(1)
            """
        )

        result = self.agent(executable).run("plan", self.root, self.root / "artifacts")

        self.assertEqual(result.exit_code, 1)
        self.assertIn("failed", result.stderr_tail)

    def test_silent_timeout_interrupts_process_group_and_is_bounded(self) -> None:
        executable = self.executable(
            """
            import signal, time
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            time.sleep(10)
            """
        )

        result = self.agent(executable, timeout=1, interrupt_grace_seconds=0.1).run(
            "plan", self.root, self.root / "artifacts"
        )

        self.assertTrue(result.timed_out)
        self.assertEqual(result.exit_code, 124)

    def test_stderr_tail_is_bounded(self) -> None:
        executable = self.executable(
            """
            import sys
            sys.stderr.write('x' * 50000)
            sys.stderr.flush()
            """
        )

        result = self.agent(executable, stderr_tail_bytes=100).run(
            "plan", self.root, self.root / "artifacts"
        )

        self.assertLessEqual(len(result.stderr_tail.encode()), 100)
        self.assertEqual(result.stderr_tail, "x" * 100)

    def test_agent_commit_is_rejected_without_reset(self) -> None:
        repository = str(self.root)
        executable = self.executable(
            f"""
            import subprocess, sys
            subprocess.run(['git', '-C', {repository!r}, 'add', '-A'], check=True)
            subprocess.run(['git', '-C', {repository!r}, 'commit', '-qm', 'forbidden'], check=True)
            """
        )

        with self.assertRaisesRegex(AgentCommittedError, "AGENT_COMMITTED"):
            self.agent(executable).run("plan", self.root, self.root / "artifacts")
        self.assertNotEqual(
            run_git(self.root, "rev-parse", "HEAD").stdout.strip(), self.base_sha
        )


if __name__ == "__main__":
    unittest.main()

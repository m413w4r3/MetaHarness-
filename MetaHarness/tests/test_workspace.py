import os
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness.models import WorkspaceSetupCommand
from metaharness.workspace import WorkspaceSetupError, prepare_workspace


def git(path: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True)


class WorkspaceSetupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        git(self.root, "init", "-q")
        git(self.root, "config", "user.name", "test")
        git(self.root, "config", "user.email", "test@example.invalid")
        (self.root / "README.md").write_text("base\n", encoding="utf-8")
        git(self.root, "add", "README.md")
        git(self.root, "commit", "-qm", "base")
        (self.root / "fake").write_text(
            "#!/usr/bin/env python3\n" + textwrap.dedent("""
            import pathlib, sys
            pathlib.Path('node_modules').mkdir(exist_ok=True)
            pathlib.Path('node_modules/cache').write_text('ok')
            print('TOKEN=env-file-secret-value')
            """),
            encoding="utf-8",
        )
        self.root.joinpath("fake").chmod(self.root.joinpath("fake").stat().st_mode | stat.S_IXUSR)
        (self.root / ".gitignore").write_text("node_modules/\n", encoding="utf-8")
        git(self.root, "add", ".")
        git(self.root, "commit", "-qm", "test setup")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_ignored_creation_is_accepted_and_logs_are_redacted(self) -> None:
        command = WorkspaceSetupCommand("deps", (str(self.root / "fake"),))
        results = prepare_workspace(
            self.root,
            (command,),
            environment=os.environ,
            artifacts_dir=self.root.parent / "artifacts",
            secrets=("env-file-secret-value",),
        )
        self.assertEqual(results[0].exit_code, 0)
        self.assertEqual((self.root.parent / "artifacts/setup/deps.stdout.log").read_text(), "TOKEN=[REDACTED]\n")

    def test_tracked_mutation_stops(self) -> None:
        script = self.root / "mutate"
        script.write_text("#!/bin/sh\necho changed >> README.md\n", encoding="utf-8")
        script.chmod(script.stat().st_mode | stat.S_IXUSR)
        git(self.root, "add", "mutate")
        git(self.root, "commit", "-qm", "add setup command")
        with self.assertRaisesRegex(WorkspaceSetupError, "WORKSPACE_SETUP_MUTATED"):
            prepare_workspace(
                self.root,
                (WorkspaceSetupCommand("mutate", (str(script),)),),
                environment=os.environ,
                artifacts_dir=self.root.parent / "artifacts",
                secrets=(),
            )


if __name__ == "__main__":
    unittest.main()

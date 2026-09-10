import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness.context import (  # noqa: E402
    ContextBundle,
    build_context,
    render_context,
)
from metaharness.gitops import current_head  # noqa: E402
from metaharness.models import ContextConfig  # noqa: E402


def run_git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
        check=True,
        shell=False,
    )


class ContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.repo = root / "repo with spaces"
        self.repo.mkdir()
        run_git(self.repo, "init")
        run_git(self.repo, "config", "user.name", "MetaHarness Tests")
        run_git(self.repo, "config", "user.email", "tests@example.invalid")

        (self.repo / "AGENTS.md").write_text("root rules\n", encoding="utf-8")
        (self.repo / "README.md").write_text("project readme\n", encoding="utf-8")
        (self.repo / "backend").mkdir()
        (self.repo / "backend" / "AGENTS.md").write_text(
            "backend rules\n", encoding="utf-8"
        )
        (self.repo / "backend" / "src").mkdir()
        self.source = self.repo / "backend" / "src" / "foo.py"
        self.source.write_text(
            "first base\nsecond base\nthird base\n", encoding="utf-8"
        )
        run_git(self.repo, "add", "--all")
        run_git(self.repo, "commit", "-m", "base")
        self.base_sha = current_head(self.repo)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def locator(self, payload: object, *, exit_code: int = 0) -> Path:
        path = Path(self.tempdir.name) / f"locator-{len(list(Path(self.tempdir.name).glob('locator-*')))}.py"
        path.write_text(
            "#!/usr/bin/env python3\n"
            "import json, pathlib, sys\n"
            "marker_arg = next((item for item in sys.argv if item.startswith('--marker=')), None)\n"
            "marker = pathlib.Path(marker_arg[9:]) if marker_arg else None\n"
            "if marker: marker.write_text('called', encoding='utf-8')\n"
            f"print({json.dumps(json.dumps(payload))})\n"
            f"raise SystemExit({exit_code})\n",
            encoding="utf-8",
        )
        path.chmod(0o755)
        return path

    def config(self, locator: Path | None = None, **kwargs: object) -> ContextConfig:
        argv = (str(locator), "query", "{query}") if locator else ()
        values: dict[str, object] = dict(
            always_files=("AGENTS.md", "README.md"),
            locator_argv=argv,
            max_hits=8,
            max_bytes=160_000,
        )
        values.update(kwargs)
        return ContextConfig(**values)

    def test_always_snippet_and_nested_agents_are_pinned_to_base(self) -> None:
        locator = self.locator(
            [
                {
                    "path": "backend/src/foo.py",
                    "start": 2,
                    "end": 3,
                    "symbol": "Foo.run",
                    "body": "THIS BODY MUST NOT BE USED",
                },
                {
                    "path": "backend/src/foo.py",
                    "start": 2,
                    "end": 2,
                    "symbol": None,
                },
            ]
        )
        self.source.write_text("first working tree\nsecond working tree\n", encoding="utf-8")

        bundle = build_context(
            self.repo,
            base_ref=self.base_sha,
            spec="show Foo",
            config=self.config(locator),
        )

        self.assertIsInstance(bundle, ContextBundle)
        self.assertTrue(bundle.locator_used)
        self.assertEqual(bundle.base_sha, self.base_sha)
        self.assertEqual(
            [path for path, _ in bundle.instruction_files],
            ["AGENTS.md", "backend/AGENTS.md", "README.md"],
        )
        self.assertEqual(bundle.excerpts[0].content, "second base\nthird base\n")
        self.assertNotIn("THIS BODY MUST NOT BE USED", render_context(bundle))
        rendered = render_context(bundle)
        self.assertIn("BASE SHA: " + self.base_sha, rendered)
        self.assertIn("### SOURCE: backend/src/foo.py:2-3", rendered)

    def test_duplicate_instructions_are_deduplicated(self) -> None:
        locator = self.locator(
            [
                {"path": "backend/src/foo.py", "start": 1, "end": 1},
                {"path": "backend/src/foo.py", "start": 2, "end": 2},
            ]
        )
        bundle = build_context(self.repo, self.base_sha, "x", self.config(locator))
        paths = [path for path, _ in bundle.instruction_files]
        self.assertEqual(paths.count("AGENTS.md"), 1)
        self.assertEqual(paths.count("backend/AGENTS.md"), 1)

    def test_unsafe_paths_and_invalid_ranges_are_rejected(self) -> None:
        locator = self.locator(
            [
                {"path": "../secret", "start": 1, "end": 1},
                {"path": "/etc/passwd", "start": 1, "end": 1},
                {"path": "backend/src/foo.py", "start": 0, "end": 1},
                {"path": "backend/src/foo.py", "start": 2, "end": 1},
                {"path": "backend/src/foo.py", "start": 1, "end": 201},
            ]
        )
        bundle = build_context(self.repo, self.base_sha, "x", self.config(locator))
        self.assertEqual(bundle.excerpts, ())
        self.assertIn("unsafe path", bundle.locator_warning or "")

    def test_locator_failure_keeps_always_files(self) -> None:
        locator = self.locator([], exit_code=7)
        bundle = build_context(self.repo, self.base_sha, "x", self.config(locator))
        self.assertTrue(bundle.locator_used)
        self.assertIn("exit status 7", bundle.locator_warning or "")
        self.assertEqual([path for path, _ in bundle.instruction_files], ["AGENTS.md", "README.md"])

    def test_head_mismatch_does_not_launch_locator(self) -> None:
        marker = Path(self.tempdir.name) / "must-not-be-called"
        locator = self.locator([],)
        # The first argument is a marker consumed by the fake executable.
        locator_argv = (str(locator), f"--marker={marker}", "{query}")
        (self.repo / "new.txt").write_text("new head\n", encoding="utf-8")
        run_git(self.repo, "add", "new.txt")
        run_git(self.repo, "commit", "-m", "new head")

        bundle = build_context(
            self.repo,
            base_ref=self.base_sha,
            spec="x",
            config=self.config(locator, locator_argv=locator_argv),
        )
        self.assertFalse(bundle.locator_used)
        self.assertFalse(marker.exists())
        self.assertIn("HEAD differs", bundle.locator_warning or "")
        self.assertEqual(bundle.base_sha, self.base_sha)

    def test_budget_omits_complete_elements(self) -> None:
        bundle = build_context(
            self.repo,
            base_ref=self.base_sha,
            spec="x",
            config=self.config(max_bytes=len("root rules\n".encode("utf-8"))),
        )
        self.assertEqual(bundle.total_bytes, len("root rules\n".encode("utf-8")))
        self.assertEqual([path for path, _ in bundle.instruction_files], ["AGENTS.md"])
        self.assertIn("README.md", bundle.omitted)


if __name__ == "__main__":
    unittest.main()

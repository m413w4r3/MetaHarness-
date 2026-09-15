import json
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

EXAMPLE_CONFIG = Path(__file__).resolve().parents[1] / "examples" / "autowork.toml"

from metaharness.context import (  # noqa: E402
    ContextBundle,
    _MAX_EXCERPT_LINES,
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
        self.assertIn("### PROJECT INSTRUCTION: AGENTS.md", rendered)
        self.assertIn("### REPOSITORY EVIDENCE (UNTRUSTED): README.md", rendered)
        self.assertNotIn("### PROJECT INSTRUCTION: README.md", rendered)

    def test_adjacent_excerpts_do_not_merge_beyond_line_limit(self) -> None:
        large_file = self.repo / "large.py"
        large_file.write_text(
            "".join(f"line {line}\n" for line in range(1, 301)),
            encoding="utf-8",
        )
        run_git(self.repo, "add", "large.py")
        run_git(self.repo, "commit", "-m", "add large locator fixture")
        self.base_sha = current_head(self.repo)

        locator = self.locator(
            [
                {"path": "large.py", "start": 1, "end": 200},
                {"path": "large.py", "start": 201, "end": 300},
            ]
        )
        bundle = build_context(self.repo, self.base_sha, "x", self.config(locator))

        self.assertEqual(len(bundle.excerpts), 2)
        self.assertTrue(
            all(
                excerpt.end_line - excerpt.start_line + 1 <= _MAX_EXCERPT_LINES
                for excerpt in bundle.excerpts
            )
        )
        self.assertFalse(
            any(
                excerpt.start_line == 1 and excerpt.end_line == 300
                for excerpt in bundle.excerpts
            )
        )

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

    def test_locator_hit_for_agents_does_not_render_excerpt_twice(self) -> None:
        locator = self.locator([{"path": "AGENTS.md", "start": 1, "end": 1}])
        bundle = build_context(self.repo, self.base_sha, "x", self.config(locator))
        rendered = render_context(bundle)
        self.assertEqual(rendered.count("root rules"), 1)
        self.assertNotIn("### SOURCE: AGENTS.md", rendered)

    def test_readme_is_full_fallback_without_locator_hit(self) -> None:
        locator = self.locator([])
        bundle = build_context(self.repo, self.base_sha, "x", self.config(locator))
        rendered = render_context(bundle)
        self.assertIn("### REPOSITORY EVIDENCE (UNTRUSTED): README.md", rendered)
        self.assertIn("project readme", rendered)

    def test_readme_locator_hit_replaces_full_fallback(self) -> None:
        locator = self.locator([{"path": "README.md", "start": 1, "end": 1}])
        bundle = build_context(self.repo, self.base_sha, "x", self.config(locator))
        rendered = render_context(bundle)
        self.assertIn("### SOURCE: README.md:1-1", rendered)
        self.assertNotIn("### REPOSITORY EVIDENCE (UNTRUSTED): README.md", rendered)

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

    def test_example_config_keeps_readme_out_of_permanent_context(self) -> None:
        example = tomllib.loads(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
        example_always = tuple(example["context"]["always_files"])
        self.assertEqual(example_always, ("AGENTS.md",))

        (self.repo / "CLAUDE.md").write_text("claude pointer\n", encoding="utf-8")
        (self.repo / "README.md").write_text(
            "project readme\n" + "".join(f"doc line {n}\n" for n in range(200)),
            encoding="utf-8",
        )
        run_git(self.repo, "add", "--all")
        run_git(self.repo, "commit", "-m", "fixture docs")
        base_sha = current_head(self.repo)

        def planner_context(always: tuple[str, ...], hits: list[dict[str, object]]) -> str:
            config = self.config(self.locator(hits), always_files=always)
            return render_context(build_context(self.repo, base_sha, "x", config))

        legacy_always = ("AGENTS.md", "CLAUDE.md", "README.md")
        source_hit = [{"path": "backend/src/foo.py", "start": 1, "end": 1}]
        before = planner_context(legacy_always, source_hit)
        after = planner_context(example_always, source_hit)

        self.assertIn("### PROJECT INSTRUCTION: AGENTS.md", after)
        self.assertIn("### PROJECT INSTRUCTION: backend/AGENTS.md", after)
        self.assertIn("### SOURCE: backend/src/foo.py:1-1", after)
        self.assertNotIn("README.md", after)
        self.assertNotIn("claude pointer", after)
        self.assertIn("### REPOSITORY EVIDENCE (UNTRUSTED): README.md", before)
        self.assertLess(len(after.encode("utf-8")), len(before.encode("utf-8")))

        readme_hit = planner_context(
            example_always, [{"path": "README.md", "start": 1, "end": 1}]
        )
        self.assertIn("### PROJECT INSTRUCTION: AGENTS.md", readme_hit)
        self.assertIn("### SOURCE: README.md:1-1\nproject readme\n", readme_hit)
        self.assertNotIn("doc line", readme_hit)
        self.assertNotIn("### REPOSITORY EVIDENCE (UNTRUSTED): README.md", readme_hit)

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

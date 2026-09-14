from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness.diagnostics import (  # noqa: E402
    MAX_ARTIFACT_BYTES,
    MAX_PLANNER_REQUEST_BYTES,
    MAX_REPORT_BYTES,
    build_run_diagnostics,
    _read_bounded,
    _summarized_events,
    write_run_diagnostics,
)
from metaharness.models import AgentConfig, ContextConfig, HarnessConfig, LLMEndpointConfig  # noqa: E402
from metaharness.state import RunStateStore  # noqa: E402
from metaharness.run_options import RunOptions, write_run_options  # noqa: E402
from metaharness.web.api import _artifact_path, get_run  # noqa: E402
from metaharness.web.pages import render_run  # noqa: E402


class DiagnosticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.runs = self.root / "runs"
        self.runs.mkdir()
        endpoint = LLMEndpointConfig("https://example.invalid", "/chat", "model", api_key_env="META_TEST_KEY")
        self.config = HarnessConfig(
            repo=self.root,
            base_ref="main",
            runs_root=self.runs,
            worktrees_root=self.root / "worktrees",
            require_clean_base=False,
            planner=endpoint,
            reviewer=endpoint,
            context=ContextConfig(),
            agent=AgentConfig(),
            checks=(),
            allow_no_required_checks=True,
            runtime_environment={"META_TEST_KEY": "test-secret-value-123"},
        )
        self.run_dir = self.runs / "diagnostic-run"
        self.run_dir.mkdir()
        store = RunStateStore(self.run_dir / "state.json")
        store.initialize("diagnostic-run")
        store.update(
            status="failed",
            base_sha="a" * 40,
            failure={"reason": "PLANNER_OUTPUT_INVALID", "detail": "bad planner output"},
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_bounded_redacted_diff_hash_and_safe_events(self) -> None:
        (self.run_dir / "spec.md").write_text("Implement the feature", encoding="utf-8")
        (self.run_dir / "planner.request.txt").write_text("prompt", encoding="utf-8")
        (self.run_dir / "agent.stderr.log").write_text(
            "Authorization: Bearer test-secret-value-123\napi_key=test-secret-value-123\n",
            encoding="utf-8",
        )
        (self.run_dir / "agent.result.json").write_text("{}", encoding="utf-8")
        events = self.run_dir / "agent.events.jsonl"
        events.write_text(json.dumps({
            "type": "item.started",
            "item": {"type": "command_execution", "command": ["bash", "--secret-arg"]},
            "arguments": {"secret": "--secret-arg"},
        }) + "\n", encoding="utf-8")
        diff = "diff --git a/a b/a\n" + ("x" * 20)
        (self.run_dir / "diff.patch").write_text(diff, encoding="utf-8")
        report = build_run_diagnostics(self.config, self.run_dir)
        self.assertIn("# MetaHarness Run Diagnostics\n", report)
        self.assertIn("PLANNER_OUTPUT_INVALID", report)
        self.assertIn("[REDACTED]", report)
        self.assertNotIn("test-secret-value-123", report)
        self.assertNotIn("--secret-arg", report)
        digest = hashlib.sha256(diff.encode()).hexdigest()
        self.assertIn(f"Size: {len(diff.encode())} bytes", report)
        self.assertIn(f"SHA256: {digest}", report)
        self.assertIn("Diff content omitted from consolidated diagnostics.", report)

    def test_oversized_artifact_and_report_are_bounded_and_deterministic(self) -> None:
        (self.run_dir / "spec.md").write_text("S\n", encoding="utf-8")
        size = MAX_PLANNER_REQUEST_BYTES + 100
        (self.run_dir / "planner.request.txt").write_text("p" * size, encoding="utf-8")
        first = build_run_diagnostics(self.config, self.run_dir)
        second = build_run_diagnostics(self.config, self.run_dir)
        normalize = lambda value: re.sub(r'"generated_at": "[^"]+"', '"generated_at": "<time>"', value)
        self.assertEqual(normalize(first), normalize(second))
        self.assertIn(f"[TRUNCATED: original {size} bytes]", first)
        self.assertLessEqual(len(first.encode("utf-8")), MAX_REPORT_BYTES)

    def test_write_and_web_view_use_the_allowlisted_report(self) -> None:
        path = write_run_diagnostics(self.config, self.run_dir)
        self.assertEqual(path.name, "diagnostics.md")
        payload = get_run(self.runs, "diagnostic-run", config=self.config)
        self.assertEqual(payload["diagnostics"]["path"], "diagnostics.md")
        page = render_run(payload, config=self.config)
        self.assertIn("DIAGNOSTICS", page)
        self.assertIn("MetaHarness Run Diagnostics", page)

    def test_artifact_lookup_stays_allowlisted(self) -> None:
        self.assertEqual(_artifact_path(self.run_dir, "diagnostics.md"), self.run_dir / "diagnostics.md")
        with self.assertRaises(ValueError):
            _artifact_path(self.run_dir, "secret.txt")

    def test_recent_events_are_read_from_the_tail(self) -> None:
        events = self.run_dir / "long.events.jsonl"
        old = json.dumps({"type": "old_failure", "item": {"type": "agent_message", "text": "OLD"}})
        latest = json.dumps({"type": "error", "item": {"type": "agent_message", "text": "LATEST FAILURE"}})
        events.write_text(old + "\n" + ("x" * (256 * 1024)) + "\n" + latest + "\n", encoding="utf-8")
        summary = _summarized_events(events, ())
        self.assertIn("LATEST FAILURE", summary)

    def test_redaction_covers_head_and_tail_boundaries(self) -> None:
        secret = "boundary-secret-value"
        limit = 64
        head = self.run_dir / "head.txt"
        head.write_bytes((b"a" * (limit - len(secret) // 2)) + secret.encode() + b" tail")
        head_text, _size, _truncated = _read_bounded(head, limit, (secret,))
        self.assertNotIn(secret, head_text)
        self.assertNotIn(secret[: len(secret) // 2], head_text)

        tail = self.run_dir / "tail.txt"
        suffix = b"s" * (limit - (len(secret) - len(secret) // 2))
        tail.write_bytes(b"p" * 20 + b"b" * 10 + secret.encode() + suffix)
        tail_text, _size, _truncated = _read_bounded(tail, limit, (secret,), tail=True)
        self.assertNotIn(secret, tail_text)
        self.assertNotIn(secret[len(secret) // 2 :], tail_text)

    def test_claude_enabled_without_artifact_is_not_reported_disabled(self) -> None:
        options = RunOptions(
            schema_version=1, protocol="v2", decomposition="balanced",
            execution_mode_policy="auto", single_step_max_mutable_paths=4,
            staged_step_max_mutable_paths=6, claude_revision_enabled=True,
            repair_cycles=0, planner_profile="planner", default_implementer_profile="impl",
            reviewer_profile="review", reviser_profile="reviser", repair_profile=None,
        )
        write_run_options(self.run_dir, options)
        report = build_run_diagnostics(self.config, self.run_dir)
        self.assertIn("Claude revision enabled by run options, but no revision artifact was produced/reached.", report)
        self.assertNotIn("Claude revision disabled for this run.", report)

    def test_prompt_footprint_is_deterministic_and_reports_usage(self) -> None:
        (self.run_dir / "spec.md").write_text("spec", encoding="utf-8")
        (self.run_dir / "context.txt").write_text("context", encoding="utf-8")
        (self.run_dir / "planner.request.txt").write_text("planner request", encoding="utf-8")
        (self.run_dir / "planner.usage.json").write_text('{"input_tokens": 21}\n', encoding="utf-8")
        step = self.run_dir / "steps" / "S01"
        step.mkdir(parents=True)
        (step / "agent.prompt.txt").write_text("step request", encoding="utf-8")
        (step / "step.json").write_text('{"usage": {"input_tokens": 7}}\n', encoding="utf-8")
        first = build_run_diagnostics(self.config, self.run_dir)
        second = build_run_diagnostics(self.config, self.run_dir)
        normalize = lambda value: re.sub(r'"generated_at": "[^"]+"', '"generated_at": "<time>"', value)
        self.assertEqual(normalize(first), normalize(second))
        self.assertIn("PROMPT FOOTPRINT", first)
        self.assertIn("| planner.request.txt | 15 |", first)
        self.assertIn("| steps/S01/agent.prompt.txt | 12 |", first)
        self.assertIn("| planner.request.txt | 15 |", first)
        self.assertLess(first.index("planner.request.txt"), first.index("steps/S01/agent.prompt.txt"))


if __name__ == "__main__":
    unittest.main()

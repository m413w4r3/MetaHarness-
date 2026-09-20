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
    _scope_violation_recovery_artifact,
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

    def test_repair_scope_policy_reports_historical_default(self) -> None:
        report = build_run_diagnostics(self.config, self.run_dir)
        self.assertIn("durable run option: historical/missing", report)
        self.assertIn("effective policy: deny-expansion", report)
        self.assertIn("max added paths: 4", report)
        self.assertIn("source: historical-default", report)

    def _publish_second_check_repair(self, scope: dict) -> None:
        directory = self.run_dir / "revision" / "check-repair-expanded" / "C01"
        directory.mkdir(parents=True)
        (directory / "scope.json").write_text(json.dumps(scope), encoding="utf-8")

    def test_a_same_scope_second_repair_is_not_reported_as_an_expansion(self) -> None:
        """"expanded check repair" is misleading when no path was added."""

        self._publish_second_check_repair({
            "schema_version": 2,
            "base_mutable_scope": ["src/service.py"],
            "added_paths": [],
            "effective_mutable_scope": ["src/service.py"],
            "policy": "deny-expansion",
            "bound": 4,
            "source": "bounded same-scope retry",
        })
        report = build_run_diagnostics(self.config, self.run_dir)
        self.assertIn("SECOND AUTOMATIC CHECK REPAIR", report)
        self.assertIn("second bounded check repair attempted", report)
        self.assertIn("scope expanded: no", report)
        self.assertIn("source: bounded same-scope retry", report)
        self.assertIn("added paths: 0", report)

    def test_b_a_true_expansion_still_reports_its_added_paths(self) -> None:
        self._publish_second_check_repair({
            "schema_version": 2,
            "base_mutable_scope": ["src/service.py"],
            "added_paths": ["tests/test_service.py"],
            "effective_mutable_scope": ["src/service.py", "tests/test_service.py"],
            "policy": "auto-bounded",
            "bound": 4,
            "source": "auto-bounded failing-test evidence",
        })
        report = build_run_diagnostics(self.config, self.run_dir)
        self.assertIn("scope expanded: yes", report)
        self.assertIn("added paths: 1", report)
        self.assertIn("- tests/test_service.py", report)

    def test_c_the_run_page_says_whether_the_scope_changed(self) -> None:
        store = RunStateStore(self.run_dir / "state.json")
        store.update(status="failed", check_repair={
            "attempted": True,
            "failure_ids": ["CHECK_FAILED:lint"],
            "second_check_repair_attempted": True,
            "scope_expanded": False,
            "added_paths": [],
        })
        page = render_run(get_run(self.runs, "diagnostic-run", config=self.config))
        self.assertIn("second bounded check repair attempted", page)
        self.assertIn("scope unchanged", page)
        self.assertNotIn("Added test paths", page)

        store.update(status="failed", check_repair={
            "attempted": True,
            "failure_ids": ["CHECK_FAILED:test"],
            "second_check_repair_attempted": True,
            "scope_expanded": True,
            "added_paths": ["tests/test_service.py"],
        })
        page = render_run(get_run(self.runs, "diagnostic-run", config=self.config))
        self.assertIn("scope expanded", page)
        self.assertIn("tests/test_service.py", page)

    def _publish_scope_repair(self, *, added: list[str] | None = None) -> None:
        directory = self.run_dir / "scope-repair" / "C01"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "scope_delta.json").write_text(
            json.dumps({"added_paths": added if added is not None else ["tests/test_service.py"]}),
            encoding="utf-8",
        )
        (directory / "scope.json").write_text(
            json.dumps({"policy": "auto-bounded"}), encoding="utf-8"
        )

    def _publish_recovery(self, source: str, *, outside: list[str] | None = None) -> None:
        directory = self.run_dir / "revision" / source / "C01"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "scope_violation_recovery.json").write_text(
            json.dumps({
                "restored_paths": ["src/service.py"],
                "outside_scope_paths": outside if outside is not None else ["tests/test_service.py"],
            }),
            encoding="utf-8",
        )

    def test_d_a_normal_check_repair_recovery_names_its_source(self) -> None:
        self._publish_scope_repair()
        self._publish_recovery("check-repair")
        report = build_run_diagnostics(self.config, self.run_dir)

        self.assertIn("CHECK REPAIR SCOPE ESCALATION", report)
        self.assertIn("trigger: REVISION_SCOPE_VIOLATION", report)
        self.assertIn("source attempt: check-repair\n", report)
        self.assertIn("failed attempt rolled back: YES", report)
        self.assertIn("observed outside-scope paths: 1", report)
        self.assertIn("scope added: 1", report)
        self.assertIn("- tests/test_service.py", report)

    def test_e_an_expanded_check_repair_recovery_is_found_and_named(self) -> None:
        """The violation happened in the second pass; nothing is in the first."""

        self._publish_scope_repair()
        self._publish_recovery("check-repair-expanded")
        self.assertFalse(
            (self.run_dir / "revision/check-repair/C01/scope_violation_recovery.json").exists()
        )
        report = build_run_diagnostics(self.config, self.run_dir)

        self.assertIn("source attempt: check-repair-expanded", report)
        self.assertIn("failed attempt rolled back: YES", report)
        self.assertIn("observed outside-scope paths: 1", report)

    def test_f_two_recovery_artifacts_are_reported_as_ambiguous(self) -> None:
        """No artifact is invented as authoritative when both exist."""

        self._publish_scope_repair()
        self._publish_recovery("check-repair", outside=[])
        self._publish_recovery("check-repair-expanded", outside=["tests/test_service.py"])
        report = build_run_diagnostics(self.config, self.run_dir)

        self.assertIn("source attempt: ambiguous", report)
        self.assertIn("no authoritative recovery artifact selected", report)
        self.assertIn("ambiguous recovery artifacts:", report)
        self.assertIn(
            "- revision/check-repair/C01/scope_violation_recovery.json"
            " (outside-scope paths: 0)",
            report,
        )
        self.assertIn(
            "- revision/check-repair-expanded/C01/scope_violation_recovery.json"
            " (outside-scope paths: 1)",
            report,
        )
        self.assertNotIn("source attempt: check-repair\n", report)
        self.assertNotIn("observed outside-scope paths: 0", report)

    def test_g_a_missing_recovery_artifact_is_reported_unavailable(self) -> None:
        self._publish_scope_repair(added=[])
        report = build_run_diagnostics(self.config, self.run_dir)

        self.assertIn("source attempt: unavailable", report)
        self.assertIn("failed attempt rolled back: NO", report)
        self.assertIn("observed outside-scope paths: 0", report)

    def test_h_the_recovery_helper_never_chooses_between_two_artifacts(self) -> None:
        self.assertEqual(_scope_violation_recovery_artifact(self.run_dir, 1), ({}, None))
        self._publish_recovery("check-repair-expanded")
        payload, source = _scope_violation_recovery_artifact(self.run_dir, 1)
        self.assertEqual(source, "check-repair-expanded")
        self.assertEqual(payload["restored_paths"], ["src/service.py"])
        self._publish_recovery("check-repair")
        payload, source = _scope_violation_recovery_artifact(self.run_dir, 1)
        self.assertEqual(source, "ambiguous")
        self.assertEqual(
            [entry["artifact"] for entry in payload["candidates"]],
            [
                "revision/check-repair/C01/scope_violation_recovery.json",
                "revision/check-repair-expanded/C01/scope_violation_recovery.json",
            ],
        )
        # A read-only projection never edits the evidence it reports on.
        self.assertTrue(
            (self.run_dir / "revision/check-repair/C01/scope_violation_recovery.json").is_file()
        )

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

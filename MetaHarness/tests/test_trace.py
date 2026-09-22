from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from metaharness.agent.base import AgentRunResult
from metaharness.models import ExecutionRole, ModelProfile, ProfileDriver, SelectionMode
from metaharness.orchestrator import Orchestrator
from metaharness.trace import TraceEvent, TraceStream


class _FailingObserver:
    def emit(self, event: TraceEvent) -> None:
        del event
        raise RuntimeError("observer unavailable")


class TraceTests(unittest.TestCase):
    def test_jsonl_is_valid_secret_free_and_resume_continues_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            first = TraceStream(run_dir, "run-1", 2, secrets=("TOP-SECRET",))
            first.emit(
                "run.created",
                phase="run",
                data={
                    "credential": "TOP-SECRET",
                    "api_key_env": "OPENAI_API_KEY",
                    "session": {"driver": "codex", "provider": "openai", "model": "gpt-test"},
                },
            )
            first.emit("checks.completed", phase="validation", data={"passed": False})

            resumed = TraceStream(run_dir, "run-1", 2)
            resumed.emit("run.failed", phase="run", data={"reason": "CHECK_FAILED"})

            path = run_dir / "trace" / "events.v1.jsonl"
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["sequence"] for row in rows], [1, 2, 3])
            self.assertEqual(rows[0]["data"]["api_key_env"], "OPENAI_API_KEY")
            self.assertNotIn("TOP-SECRET", path.read_text(encoding="utf-8"))
            for row in rows:
                self.assertEqual(row["schema_version"], 1)
                self.assertEqual(row["run_id"], "run-1")
                self.assertEqual(row["pipeline_version"], 2)

    def test_external_observer_failure_does_not_prevent_local_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stream = TraceStream(
                Path(directory), "run-2", 2, sink=_FailingObserver()
            )
            stream.emit("run.created")
            self.assertTrue((Path(directory) / "trace/events.v1.jsonl").is_file())

    def test_diff_reference_and_session_keep_unavailable_metrics_null(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "diff.patch"
            artifact.write_text("diff --git a/a b/a\n", encoding="utf-8")
            stream = TraceStream(Path(directory), "run-3", 2)
            event = stream.emit(
                "step.committed",
                data={
                    "parent_sha": "parent",
                    "commit_sha": "commit",
                    "tree_sha": "tree",
                    "changed_paths": ["a"],
                    "diff_artifact": str(artifact),
                    "diff_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                },
            )
            self.assertEqual(event.as_dict()["data"]["diff_artifact"], str(artifact))
            self.assertEqual(len(event.as_dict()["data"]["diff_sha256"]), 64)

        profile = ModelProfile(
            id="codex-profile",
            display_name="Codex",
            roles=(ExecutionRole.IMPLEMENTER,),
            driver=ProfileDriver.CODEX,
            model="gpt-test",
            selection_mode=SelectionMode.CLI,
            effort="high",
            provider="openai",
            driver_version="local-test-driver",
        )
        owner = object.__new__(Orchestrator)
        owner.config = SimpleNamespace(
            agent=SimpleNamespace(env_allowlist=()),
            codex_runtime=SimpleNamespace(home=None),
            claude_runtime=SimpleNamespace(home=None),
        )
        session = owner._trace_session(
            profile=profile,
            selected=SimpleNamespace(
                profile_id="codex-profile",
                config_sha256="profile-fingerprint",
                provider="openai",
                model="gpt-test",
                driver="codex",
                driver_version="local-test-driver",
                effort="high",
            ),
            role=ExecutionRole.IMPLEMENTER,
            prompt_bytes=12,
            started_at="2026-01-01T00:00:00Z",
            started_mono=0.0,
            result=AgentRunResult(
                status="completed",
                exit_reason=None,
                tree_before="tree-before",
                tree_after="tree-after",
                usage=None,
                external_session_id=None,
                report_path=None,
                driver="codex",
            ),
        )
        self.assertEqual(session["driver"], "codex")
        self.assertEqual(session["provider"], "openai")
        self.assertEqual(session["model"], "gpt-test")
        self.assertEqual(session["profile_fingerprint"], "profile-fingerprint")
        self.assertEqual(session["driver_version"], "local-test-driver")
        self.assertIsNone(session["input_tokens"])
        self.assertIsNone(session["cached_input_tokens"])
        self.assertIsNone(session["output_tokens"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import hashlib
from dataclasses import replace
import sys
import tempfile
import threading
import unittest
from http.client import HTTPConnection
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unittest.mock import patch

from metaharness.approval import compute_plan_identity_from_run, write_check_authority
from metaharness.models import (
    CheckConfig,
    ContextConfig,
    HarnessConfig,
    LLMEndpointConfig,
    ModelProfile,
    ExecutionRole,
    ProfileDriver,
    RunDisposition,
    RunMachineState,
    RunPhase,
    RoutingConfig,
    SelectionMode,
    UIConfig,
)
from metaharness.run_options import RunOptions, write_run_options
from metaharness.state import RunStateStore
from metaharness.web import api
from metaharness.web.server import create_server

# One fixture state per operator-facing status the web tests render.  The
# status itself is never written: each fixture names the machine state the
# projection derives it from.
FIXTURE_STATES = {
    "created": RunMachineState(),
    "planning": RunMachineState(RunPhase.PLANNER),
    "awaiting_plan_approval": RunMachineState(RunPhase.PLAN_APPROVAL),
    "implementing": RunMachineState(RunPhase.IMPLEMENT_STEP),
    "validating": RunMachineState(RunPhase.DETERMINISTIC_GATE),
    "reviewing": RunMachineState(RunPhase.FINAL_REVIEW),
    "waiting_external": RunMachineState(disposition=RunDisposition.WAIT_EXTERNAL, reason="AGENT_TIMEOUT"),
    "committed": RunMachineState(RunPhase.CANDIDATE_PUSH, RunDisposition.COMPLETED),
}


class WebServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.runs = root / "runs"
        self.runs.mkdir()
        endpoint = LLMEndpointConfig("https://example.invalid", "/chat", "model")
        profiles = {
            "planner": ModelProfile("planner", "Planner", (ExecutionRole.PLANNER,), ProfileDriver.OPENAI_CHAT, "planner", SelectionMode.REQUEST, base_url=endpoint.base_url, endpoint_path=endpoint.endpoint_path),
            "implementer": ModelProfile("implementer", "Implementer", (ExecutionRole.IMPLEMENTER,), ProfileDriver.EXTERNAL, "worker", SelectionMode.CLI, argv=("true",)),
            "reviewer": ModelProfile("reviewer", "Reviewer", (ExecutionRole.REVIEWER,), ProfileDriver.OPENAI_CHAT, "reviewer", SelectionMode.REQUEST, base_url=endpoint.base_url, endpoint_path=endpoint.endpoint_path),
        }
        self.config = HarnessConfig(
            repo=root,
            base_ref="HEAD",
            runs_root=self.runs,
            worktrees_root=root / "worktrees",
            require_clean_base=True,
            context=ContextConfig(),
            check_catalog=(CheckConfig("lint", ("echo", "argv-secret"), description="Repository lint gate."),),
            allow_no_required_checks=True,
            runtime_environment={"API_KEY": "secret-test-value"},
            ui=UIConfig(
                default_planner_profile="planner",
                default_reviewer_profile="reviewer",
            ),
            model_profiles=profiles,
            routing=RoutingConfig(
                mechanical_profile="implementer",
                reasoning_profile="implementer",
                agentic_profile="implementer",
            ),
        )
        self.server = create_server(self.config, port=0)
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp.cleanup()

    def request(self, method: str, path: str, body: object | None = None, token: str | None = None):
        connection = HTTPConnection("127.0.0.1", self.server.server_port)
        headers = {"Accept": "application/json"}
        if token is not None:
            headers["X-MetaHarness-Token"] = token
        encoded = None
        if body is not None:
            encoded = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
            headers["Content-Length"] = str(len(encoded))
        connection.request(method, path, body=encoded, headers=headers)
        response = connection.getresponse()
        content = response.read()
        self.last_content_type = response.getheader("Content-Type", "")
        self.last_location = response.getheader("Location")
        connection.close()
        return response.status, json.loads(content) if content else None, content

    def create_run(self, run_id: str, status: str = "created") -> Path:
        run_dir = self.runs / run_id
        store = RunStateStore(run_dir / "state.json")
        store.initialize(run_id)
        store.set_run_state(FIXTURE_STATES[status], planning_protocol="v2")
        return run_dir

    def test_v1_get_routes_are_json_and_delegate_to_existing_api(self) -> None:
        with (
            patch("metaharness.web.server.model_profiles", return_value={"profiles": []}) as profiles,
            patch("metaharness.web.server.list_runs", return_value=[{"run_id": "r1"}]) as runs,
            patch("metaharness.web.server.get_run", return_value={"run_id": "r1"}) as get_run,
            patch("metaharness.web.server.live_status", return_value={"status": "created"}) as live,
            patch("metaharness.web.server.progress", return_value={"next_offset": 4}) as progress,
        ):
            cases = (
                ("/api/v1/health", 200),
                ("/api/v1/config", 200),
                ("/api/v1/model-profiles", 200),
                ("/api/v1/runs", 200),
                ("/api/v1/runs/r1", 200),
                ("/api/v1/runs/r1/live", 200),
                ("/api/v1/runs/r1/progress?offset=3", 200),
            )
            for path, expected_status in cases:
                with self.subTest(path=path):
                    status, payload, raw = self.request("GET", path)
                    self.assertEqual(status, expected_status)
                    self.assertTrue(raw)
                    self.assertIn("application/json", self.last_content_type)
                    self.assertIsNone(self.last_location)
                    self.assertIsInstance(payload, dict)
            profiles.assert_called_once_with(self.config)
            runs.assert_called_once_with(self.runs)
            get_run.assert_called_once_with(self.runs, "r1", config=self.config)
            live.assert_called_once_with(self.runs, "r1", self.config)
            progress.assert_called_once_with(self.runs, "r1", 3, self.config)

    def test_config_endpoint_exposes_effective_safe_config(self) -> None:
        status, payload, raw = self.request("GET", "/api/v1/config")
        self.assertEqual(status, 200)
        self.assertEqual(payload["repository"], {
            "repo": str(self.config.repo), "base_ref": "HEAD", "remote": "origin",
        })
        self.assertEqual(payload["planning"], {
            "protocol": self.config.planning.protocol,
            "decomposition": self.config.planning.decomposition,
            "execution_mode_policy": self.config.planning.execution_mode_policy,
            "single_step_max_mutable_paths": self.config.planning.single_step_max_mutable_paths,
            "staged_step_max_mutable_paths": self.config.planning.staged_step_max_mutable_paths,
            "max_steps_per_plan": self.config.planning.max_steps_per_plan,
        })
        self.assertEqual(payload["revision"], {
            "enabled": self.config.revision.enabled,
            "max_check_repair_attempts": self.config.revision.max_check_repair_attempts,
            "max_correction_cycles": self.config.revision.max_correction_cycles,
        })
        self.assertEqual(payload["ui"]["max_active_runs"], self.config.ui.max_active_runs)
        self.assertEqual(payload["checks"], [{"id": "lint", "description": "Repository lint gate."}])
        self.assertIsNone(payload["config_fingerprint"])
        self.assertNotIn("argv-secret", raw.decode())
        self.assertNotIn("secret-test-value", raw.decode())
        self.assertNotIn("API_KEY", raw.decode())
        self.assertLess(len(raw), 64 * 1024)

    def test_config_fingerprint_hashes_stable_original_file_bytes(self) -> None:
        source = Path(self.temp.name) / "source.toml"
        source.write_bytes(b"config = 'original'\n")
        fingerprints = []
        for _ in range(2):
            with patch("metaharness.web.server.load_config", return_value=self.config):
                server = create_server(source, port=0)
            fingerprints.append(server.config_fingerprint)
            server.server_close()
        self.assertEqual(fingerprints[0], fingerprints[1])
        self.assertEqual(fingerprints[0], hashlib.sha256(source.read_bytes()).hexdigest())

    def test_config_check_catalogue_is_bounded(self) -> None:
        original = self.server.config
        self.server.config = replace(
            original,
            check_catalog=tuple(
                CheckConfig(f"check-{index}-" + "x" * 300, ("secret-command",), description="d" * 300)
                for index in range(300)
            ),
        )
        try:
            status, payload, raw = self.request("GET", "/api/v1/config")
        finally:
            self.server.config = original
        self.assertEqual(status, 200)
        self.assertTrue(payload["checks_truncated"])
        self.assertEqual(len(payload["checks"]), 80)
        self.assertLess(len(raw), 64 * 1024)
        self.assertNotIn("secret-command", raw.decode())

    def test_v1_get_rejects_bad_offsets_and_run_ids_as_json(self) -> None:
        for path, status_expected in (
            ("/api/v1/runs/r1/progress?offset=-1", 400),
            ("/api/v1/runs/r1/progress?offset=1&offset=2", 400),
            ("/api/v1/runs/%2e%2e", 400),
        ):
            with self.subTest(path=path):
                status, payload, raw = self.request("GET", path)
                self.assertEqual(status, status_expected)
                self.assertEqual(set(payload), {"error", "message"})
                self.assertTrue(raw)

    def test_artifact_endpoint_reads_allowlisted_files_with_a_bound(self) -> None:
        run_dir = self.create_run("r1")
        (run_dir / "reviewer.raw.md").write_text("review text", encoding="utf-8")
        (run_dir / "trace").mkdir()
        (run_dir / "trace" / "events.v1.jsonl").write_text(
            "event one\nevent two\n", encoding="utf-8"
        )

        status, payload, _ = self.request(
            "GET", "/api/v1/runs/r1/artifact?name=reviewer.raw.md"
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["name"], "reviewer.raw.md")
        self.assertTrue(payload["exists"])
        self.assertEqual(payload["content"], "review text")
        self.assertFalse(payload["truncated"])
        self.assertEqual(payload["size"], len("review text"))

        status, missing, _ = self.request(
            "GET", "/api/v1/runs/r1/artifact?name=review.json"
        )
        self.assertEqual(status, 200)
        self.assertFalse(missing["exists"])
        self.assertIsNone(missing["content"])

        (run_dir / "planner.raw.md").write_text("0123456789", encoding="utf-8")
        with patch.object(api, "MAX_ARTIFACT_READ_BYTES", 5):
            status, truncated, _ = self.request(
                "GET", "/api/v1/runs/r1/artifact?name=planner.raw.md"
            )
        self.assertEqual(status, 200)
        self.assertTrue(truncated["truncated"])
        self.assertEqual(truncated["content"], "56789")
        self.assertEqual(truncated["size"], 10)

    def test_artifact_endpoint_rejects_traversal_and_encoded_names(self) -> None:
        self.create_run("r1")
        for name in (
            "../etc/passwd",
            "../../state.json",
            "%2e%2e/",
            "cycles/../../state.json",
        ):
            with self.subTest(name=name):
                status, payload, raw = self.request(
                    "GET", f"/api/v1/runs/r1/artifact?name={name}"
                )
                self.assertEqual(status, 404)
                self.assertEqual(set(payload), {"error", "message"})
                self.assertNotIn("Traceback", raw.decode("utf-8"))

        for query, expected_status in (
            ("name=%73tate.json", 404),
            ("name=state.json&name=diff.patch", 400),
            ("name=state.json&extra=1", 400),
        ):
            with self.subTest(query=query):
                status, payload, _ = self.request(
                    "GET", f"/api/v1/runs/r1/artifact?{query}"
                )
                self.assertEqual(status, expected_status)
                self.assertEqual(set(payload), {"error", "message"})

    def test_v1_post_routes_authenticate_and_return_json_without_redirects(self) -> None:
        with (
            patch("metaharness.web.server.create_run", return_value={"run_id": "new", "location": "/runs/new"}) as create,
            patch("metaharness.web.server.approve_run", return_value={"decision": "REJECT"}) as approve,
            patch("metaharness.web.server.approve_repair_scope", return_value={"decision": "APPROVE"}) as scope,
            patch("metaharness.web.server.resume_run_request", return_value={"run_id": "r1", "location": "/runs/r1"}) as resume,
            patch("metaharness.web.server.recover_plan_request", return_value={"run_id": "r1", "location": "/runs/r1"}) as recover,
        ):
            mutations = (
                ("/api/v1/runs", {"spec": "spec", "run_id": "new", "planner_profile": "planner"}),
                ("/api/v1/runs/r1/approval", {"decision": "REJECT"}),
                ("/api/v1/runs/r1/scope-approval", {"decision": "APPROVE"}),
                ("/api/v1/runs/r1/resume", {}),
                ("/api/v1/runs/r1/recover-plan", {"plan": "META PLAN v2\n"}),
            )
            for path, body in mutations:
                with self.subTest(path=path):
                    status, payload, raw = self.request("POST", path, body, self.server.browser_token)
                    self.assertIn(status, (200, 202))
                    self.assertTrue(raw)
                    self.assertIn("application/json", self.last_content_type)
                    self.assertIsNone(self.last_location)
                    self.assertIsInstance(payload, dict)
                    if status == 202:
                        self.assertTrue(payload["accepted"])
            self.assertEqual(create.call_args.kwargs["run_id"], "new")
            approve.assert_called_once()
            scope.assert_called_once_with(self.runs, "r1", "APPROVE")
            resume.assert_called_once_with(self.server.run_manager, self.runs, "r1")
            recover.assert_called_once_with(self.server.run_manager, self.runs, "r1", "META PLAN v2\n")

    def test_v1_mutations_reject_missing_token_unknown_fields_and_nonempty_resume(self) -> None:
        requests = (
            ("/api/v1/runs", {"spec": "spec"}),
            ("/api/v1/runs/r1/approval", {"decision": "REJECT"}),
            ("/api/v1/runs/r1/scope-approval", {"decision": "REJECT"}),
            ("/api/v1/runs/r1/resume", {}),
            ("/api/v1/runs/r1/recover-plan", {"plan": "x"}),
        )
        for path, body in requests:
            with self.subTest(path=path):
                status, payload, _ = self.request("POST", path, body)
                self.assertEqual(status, 403)
                self.assertEqual(set(payload), {"error", "message"})
        status, payload, _ = self.request(
            "POST", "/api/v1/runs", {"spec": "x", "arbitrary": True}, self.server.browser_token
        )
        self.assertEqual(status, 400)
        self.assertEqual(set(payload), {"error", "message"})

    def test_v1_invalid_mutation_payloads_and_unsupported_methods_stay_json(self) -> None:
        cases = (
            ("/api/v1/runs/r1/approval", {"decision": "REJECT", "unexpected": 1}),
            ("/api/v1/runs/r1/scope-approval", {"decision": "MAYBE"}),
            ("/api/v1/runs/r1/recover-plan", {"plan": 1}),
        )
        for path, body in cases:
            with self.subTest(path=path):
                status, payload, _ = self.request("POST", path, body, self.server.browser_token)
                self.assertEqual(status, 400)
                self.assertIn("application/json", self.last_content_type)
                self.assertIsNone(self.last_location)
                self.assertEqual(set(payload), {"error", "message"})
        status, payload, raw = self.request("PUT", "/api/v1/runs")
        self.assertEqual(status, 501)
        self.assertTrue(raw)
        self.assertIn("application/json", self.last_content_type)
        self.assertEqual(set(payload), {"error", "message"})
        status, payload, _ = self.request(
            "POST", "/api/v1/runs/r1/resume", {"unexpected": True}, self.server.browser_token
        )
        self.assertEqual(status, 400)
        self.assertEqual(set(payload), {"error", "message"})

    def test_list_and_run_and_missing_artifacts(self) -> None:
        run_dir = self.create_run("run-1")
        status, payload, _ = self.request("GET", "/api/runs")
        self.assertEqual(status, 200)
        self.assertEqual(payload["runs"][0]["run_id"], "run-1")
        status, payload, _ = self.request("GET", "/api/runs/run-1")
        self.assertEqual(status, 200)
        self.assertEqual(payload["state"]["run_id"], "run-1")
        self.assertIsNone(payload["plan"]["raw"])
        self.assertIsNone(payload["checks"])
        self.assertFalse((run_dir / "plan_approval.json").exists())

    def test_unknown_and_traversal_runs_are_rejected(self) -> None:
        for path in (
            "/api/runs/unknown",
            "/api/runs/../x",
            "/api/runs/a%2Fb",
            "/api/runs/%2e%2e",
        ):
            with self.subTest(path=path):
                status, _payload, _ = self.request("GET", path)
                self.assertIn(status, (400, 404))

    def test_approval_requires_token_and_is_exclusive(self) -> None:
        run_dir = self._create_v2_approval_run("waiting")
        for token in (None, "wrong"):
            status, _payload, _ = self.request(
                "POST", "/api/runs/waiting/approval", {"decision": "APPROVE"}, token
            )
            self.assertEqual(status, 403)
        status, payload, _ = self._approve_v2("waiting")
        self.assertEqual(status, 200)
        self.assertEqual(payload["decision"], "APPROVE")
        approval = json.loads((run_dir / "plan_approval.json").read_text())
        self.assertEqual(approval["source"], "web-ui")
        status, _payload, _ = self.request(
            "POST",
            "/api/runs/waiting/approval",
            {"decision": "REJECT"},
            self.server.browser_token,
        )
        self.assertEqual(status, 409)

    def _create_v2_approval_run(self, run_id: str, *, stored_checks_sha256: str | None = None) -> Path:
        run_dir = self.create_run(run_id, "awaiting_plan_approval")
        raw = "META PLAN v2\nSTATUS: READY\n"
        contract = "# v2 contract\n"
        (run_dir / "planner.raw.md").write_text(raw, encoding="utf-8")
        (run_dir / "implementation_contract.md").write_text(contract, encoding="utf-8")
        step_contract = run_dir / "steps" / "S01" / "contract.md"
        step_contract.parent.mkdir(parents=True)
        step_contract.write_text("step contract\n", encoding="utf-8")
        import hashlib

        (run_dir / "implementation_bundle.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "execution_mode": "SINGLE",
                    "required_checks": [],
                    "steps": [{
                        "id": "S01",
                        "title": "One",
                        "execution_class": "MECHANICAL",
                        "depends_on": None,
                        "contract_sha256": hashlib.sha256(step_contract.read_bytes()).hexdigest(),
                    }],
                },
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        write_check_authority(
            run_dir, [CheckConfig("lint", ("python", "-c", "pass"))],
            required_check_ids=("lint",),
        )
        options_sha256 = write_run_options(run_dir, RunOptions.from_config(self.config))
        actual = compute_plan_identity_from_run(run_dir)
        stored = actual.__dict__.copy()
        stored["checks_sha256"] = stored_checks_sha256
        RunStateStore(run_dir / "state.json").update(
            status="awaiting_plan_approval",
            planning_protocol="v2",
            plan_identity=stored,
            run_options_sha256=options_sha256,
            execution={"planner": {"profile_id": "planner"}},
        )
        return run_dir

    def _approve_v2(self, run_id: str):
        return self.request(
            "POST",
            f"/api/runs/{run_id}/approval",
            {
                "decision": "APPROVE",
                "final_reviewer_profile": "reviewer",
                "step_profile__S01": "implementer",
            },
            self.server.browser_token,
        )

    def test_v2_approval_publishes_the_complete_same_plan_identity(self) -> None:
        run_dir = self._create_v2_approval_run("v2-checks")

        status, _payload, _ = self._approve_v2("v2-checks")
        self.assertEqual(status, 200)
        approval = json.loads((run_dir / "plan_approval.json").read_text(encoding="utf-8"))
        state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
        for field in ("raw_sha256", "contract_sha256", "bundle_sha256", "execution_sha256", "checks_sha256"):
            self.assertEqual(state["plan_identity"][field], approval[field])
        self.assertIsNotNone(approval["checks_sha256"])

    def test_v2_approval_rejects_a_different_durable_checks_hash(self) -> None:
        run_dir = self._create_v2_approval_run("v2-check-conflict", stored_checks_sha256="f" * 64)

        status, _payload, _ = self._approve_v2("v2-check-conflict")
        self.assertEqual(status, 409)
        self.assertFalse((run_dir / "plan_approval.json").exists())

    def post_form(self, path: str, fields: dict[str, str]) -> int:
        from urllib.parse import urlencode

        body = urlencode(fields).encode()
        connection = HTTPConnection("127.0.0.1", self.server.server_port)
        connection.request("POST", path, body=body, headers={
            "Content-Type": "application/x-www-form-urlencoded", "Content-Length": str(len(body)),
        })
        response = connection.getresponse()
        response.read()
        connection.close()
        return response.status

    def test_approval_fields_follow_the_actual_12_step_bundle(self) -> None:
        run_dir = self.create_run("form-steps", "committed")
        RunStateStore(run_dir / "state.json").update(
            status="committed", execution={"planner": {"profile_id": "planner"}},
        )
        (run_dir / "implementation_bundle.json").write_text(
            json.dumps({"steps": [{"id": f"S{number:02d}"} for number in range(1, 13)]}),
            encoding="utf-8",
        )
        fields = {"_token": self.server.browser_token, "decision": "APPROVE", "final_reviewer_profile": "r"}
        twelve = {f"step_profile__S{number:02d}": "luna" for number in range(1, 13)}
        # The fields for the actual bundle are accepted; the gate then refuses
        # this deliberately incomplete synthetic run.
        self.assertEqual(self.post_form("/runs/form-steps/approval", {**fields, **twelve}), 409)
        self.assertEqual(
            self.post_form("/runs/form-steps/approval", {**fields, **twelve, "step_profile__S13": "luna"}), 400
        )

        page_dir = self.create_run("form-page", "awaiting_plan_approval")
        (page_dir / "implementation_bundle.json").write_text(
            json.dumps({"steps": [{"id": f"S{number:02d}"} for number in range(1, 13)]}),
            encoding="utf-8",
        )
        RunStateStore(page_dir / "state.json").update(
            status="awaiting_plan_approval",
            planning_protocol="v2",
            planner={
                "execution_mode": "STAGED",
                "steps": [{"id": f"S{number:02d}", "title": f"Step {number}"} for number in range(1, 13)],
            },
        )
        page = self.get_html("/runs/form-page")
        self.assertEqual(page.count('name="step_profile__'), 12)
        self.assertNotIn('name="step_profile__S13"', page)

    def test_wrong_state_is_conflict(self) -> None:
        self.create_run("done", "committed")
        status, _payload, _ = self.request(
            "POST", "/api/runs/done/approval", {"decision": "APPROVE"}, self.server.browser_token
        )
        self.assertEqual(status, 409)

    def test_progress_ignores_malformed_lines_and_uses_byte_offsets(self) -> None:
        run_dir = self.create_run("events")
        events = [
            "not json\n",
            json.dumps({"type": "turn.started"}) + "\n",
            json.dumps({"item": {"command": "python -m unittest"}}) + "\n",
        ]
        (run_dir / "agent.events.jsonl").write_text("".join(events), encoding="utf-8")
        status, payload, _ = self.request("GET", "/api/runs/events/progress?offset=0")
        self.assertEqual(status, 200)
        self.assertTrue(any("turn.started" in item for item in payload["events"]))
        status, second, _ = self.request(
            "GET", f"/api/runs/events/progress?offset={payload['next_offset']}"
        )
        self.assertEqual(status, 200)
        self.assertEqual(second["events"], [])

    def test_unified_progress_projects_trace_agent_and_redacts_credentials(self) -> None:
        run_dir = self.create_run("unified")
        trace = run_dir / "trace" / "events.v1.jsonl"
        trace.parent.mkdir(parents=True)
        trace.write_text(json.dumps({
            "schema_version": 1, "sequence": 1, "timestamp": "2026-09-23T15:45:17Z",
            "event": "contract_repair.waiting_external", "phase": "repair", "cycle": 1,
            "step_id": "S04", "data": {"detail": "HTTP 503 after 3 attempts", "action": "waiting external",
                "authorization": "Bearer SECRET"},
        }) + "\n", encoding="utf-8")
        agent = run_dir / "cycles/001/implementation/steps/S01/agent.events.jsonl"
        agent.parent.mkdir(parents=True)
        agent.write_text(json.dumps({"type": "item.started", "item": {"type": "command_execution", "command": "rg Authorization SECRET"}}) + "\n" + json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "DEEPSEEK_API_KEY=SECRET found issue"}}) + "\n", encoding="utf-8")
        status, first, _ = self.request("GET", "/api/runs/unified/progress?offset=0")
        self.assertEqual(status, 200)
        self.assertTrue(any("[recovery] [S04]" in item and "HTTP 503" in item for item in first["events"]))
        self.assertTrue(any("[S01] tool: command rg" in item for item in first["events"]))
        self.assertTrue(any("[REDACTED]" in item for item in first["events"]))
        self.assertNotIn("SECRET", "\n".join(first["events"]))
        status, second, _ = self.request("GET", f"/api/runs/unified/progress?offset={first['next_offset']}")
        self.assertEqual(status, 200)
        self.assertEqual(second["events"], [])
        projection = (run_dir / "progress/events.v1.jsonl").read_text(encoding="utf-8")
        self.assertNotIn("SECRET", projection)

    def test_waiting_external_live_payload_includes_durable_history(self) -> None:
        run_dir = self.create_run("waiting-progress", "waiting_external")
        (run_dir / "state.json").write_text(json.dumps({"status": "waiting_external", "current_step": None}), encoding="utf-8")
        trace = run_dir / "trace/events.v1.jsonl"
        trace.parent.mkdir(parents=True)
        trace.write_text(json.dumps({"sequence": 1, "timestamp": "2026-09-23T15:45:17Z", "event": "repair.waiting_external", "phase": "repair", "step_id": "S04", "data": {"reason": "LLM_FAILURE", "detail": "HTTP 503 after 3 attempts"}}) + "\n", encoding="utf-8")
        payload = api.live_status(self.runs, "waiting-progress", self.config)
        self.assertTrue(any("HTTP 503" in event for event in payload["progress_events"]))

    def test_progress_is_capped_to_complete_lines(self) -> None:
        run_dir = self.create_run("large-events")
        line = json.dumps({"type": "turn.completed"}) + "\n"
        (run_dir / "agent.events.jsonl").write_text(line * 40_000, encoding="utf-8")
        status, payload, _ = self.request("GET", "/api/runs/large-events/progress")
        self.assertEqual(status, 200)
        self.assertLessEqual(payload["next_offset"], 256 * 1024)
        self.assertGreater(len(payload["events"]), 0)

    def get_html(self, path: str) -> str:
        connection = HTTPConnection("127.0.0.1", self.server.server_port)
        connection.request("GET", path)
        response = connection.getresponse()
        content = response.read().decode("utf-8")
        connection.close()
        self.assertEqual(response.status, 200)
        return content

    def assert_poll_shape(self, payload: dict) -> None:
        # Fields read by the run-page polling script.
        for key in (
            "run_id",
            "status",
            "updated_at",
            "state",
            "plan",
            "checks",
            "review",
            "reviewer_raw",
            "reviewer_raw_available",
            "failure",
            "commit_sha",
        ):
            self.assertIn(key, payload)
        for key in ("status", "base_sha", "branch", "worktree", "commit_sha", "failure"):
            self.assertIn(key, payload["state"])
        self.assertIn("raw", payload["plan"])
        self.assertIn("contract", payload["plan"])
        self.assertEqual(payload["status"], payload["state"]["status"])

    def test_run_page_observes_transitions_without_manual_reload(self) -> None:
        run_dir = self.runs / "live"
        store = RunStateStore(run_dir / "state.json")
        store.initialize("live", base_sha="b" * 40, branch="metaharness/live", worktree="/tmp/wt live")
        store.set_run_state(RunMachineState(RunPhase.IMPLEMENT_STEP))

        page = self.get_html("/runs/live")
        self.assertIn("Run <span class=\"mono\">live</span>", page)
        # P29: no full-page refresh; targeted polling through the static
        # same-origin script only (no inline script).
        self.assertNotIn('http-equiv="refresh"', page)
        self.assertIn('<script src="/static/run.js" defer></script>', page)
        self.assertNotIn("<script>", page)

        status, payload, _ = self.request("GET", "/api/runs/live")
        self.assertEqual(status, 200)
        self.assert_poll_shape(payload)
        self.assertEqual(payload["status"], "implementing")
        self.assertEqual(payload["state"]["worktree"], "/tmp/wt live")

        checks = [{"name": "unit", "exit_code": 0, "timed_out": False, "stdout_tail": "<b>ok</b>"}]
        gate = run_dir / "cycles/001/checks/post-implementation"
        gate.mkdir(parents=True)
        (gate / "checks.json").write_text(json.dumps(checks), encoding="utf-8")
        store.set_run_state(RunMachineState(RunPhase.DETERMINISTIC_GATE))
        status, payload, _ = self.request("GET", "/api/runs/live")
        self.assert_poll_shape(payload)
        self.assertEqual(payload["status"], "validating")
        self.assertEqual(payload["checks"], checks)
        self.assertFalse(payload["reviewer_raw_available"])

        review = run_dir / "cycles/001/review"
        review.mkdir(parents=True)
        (review / "review.json").write_text(
            json.dumps({"verdict": "PASS", "route": "NONE", "summary": "ok"}), encoding="utf-8"
        )
        (review / "reviewer.raw.md").write_text("VERDICT: PASS\n", encoding="utf-8")
        store.set_run_state(RunMachineState(RunPhase.FINAL_REVIEW))
        status, payload, _ = self.request("GET", "/api/runs/live")
        self.assert_poll_shape(payload)
        self.assertEqual(payload["status"], "reviewing")
        self.assertEqual(payload["review"]["verdict"], "PASS")
        self.assertTrue(payload["reviewer_raw_available"])
        self.assertEqual(payload["reviewer_raw"], "VERDICT: PASS\n")

        store.set_run_state(
            RunMachineState(RunPhase.CANDIDATE_PUSH, RunDisposition.COMPLETED),
            commit_sha="c" * 40,
        )
        status, payload, _ = self.request("GET", "/api/runs/live")
        self.assert_poll_shape(payload)
        self.assertEqual(payload["status"], "committed")
        self.assertEqual(payload["state"]["commit_sha"], "c" * 40)
        self.assertIsNone(payload["failure"])

        failed = self.create_run("broken", "implementing")
        RunStateStore(failed / "state.json").record_failure("CHECK_FAILED", "unit")
        status, payload, _ = self.request("GET", "/api/runs/broken")
        self.assert_poll_shape(payload)
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["state"]["failure"], {"reason": "CHECK_FAILED", "detail": "unit"})

    def test_approval_controls_follow_status(self) -> None:
        self.create_run("before", "planning")
        page = self.get_html("/runs/before")
        self.assertNotIn('action="/runs/before/approval"', page)
        self.assertNotIn('http-equiv="refresh"', page)
        self.assertIn("/static/run.js", page)
        self.assertNotIn(self.server.browser_token, page)

        self._create_v2_approval_run("gate")
        page = self.get_html("/runs/gate")
        self.assertIn('action="/runs/gate/approval"', page)
        self.assertIn('>APPROVE PLAN</button>', page)
        self.assertIn('>REJECT PLAN</button>', page)
        self.assertNotIn('http-equiv="refresh"', page)

    def test_waiting_human_planner_is_rendered_as_waiting_not_failed(self) -> None:
        run_dir = self.runs / "human-decision"
        run_dir.mkdir()
        state = {
            "status": "waiting_human",
            "cycle": 1,
            "planner": {"decision": "BLOCKED"},
            "failure": {"reason": "SPEC_DECISION_REQUIRED", "detail": {}},
        }
        rows = api.run_pipeline(run_dir, state, self.config)
        planner = next(item for item in rows if item["key"] == "planner")
        self.assertEqual(planner["state"], "waiting")
        self.assertIn("waiting_human", api.LIVE_STOP_STATUSES)


class ProgressOffsetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.runs = Path(self.temp.name)
        (self.runs / "p").mkdir()
        self.path = self.runs / "p" / "agent.events.jsonl"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write(self, data: bytes, mode: str = "wb") -> None:
        with self.path.open(mode) as stream:
            stream.write(data)

    def poll(self, offset: int) -> dict:
        with self.path.open("rb") as stream:
            return api._read_progress(stream, min(offset, self.path.stat().st_size))

    def drain(self, offset: int = 0, max_requests: int = 200) -> tuple[int, list[str]]:
        """Poll like the browser until the offset stops, checking progress."""

        events: list[str] = []
        for _ in range(max_requests):
            payload = self.poll(offset)
            events.extend(payload["events"])
            if payload["next_offset"] == offset:
                return offset, events
            self.assertGreater(payload["next_offset"], offset)
            offset = payload["next_offset"]
        self.fail("progress offset did not settle")

    @staticmethod
    def event(kind: str) -> bytes:
        return json.dumps({"type": kind}).encode() + b"\n"

    def test_normal_lines_advance_normally(self) -> None:
        data = self.event("one") + self.event("two") + self.event("three")
        self.write(data)
        payload = self.poll(0)
        self.assertEqual(payload, {"next_offset": len(data), "events": ["one", "two", "three"]})
        self.assertEqual(self.poll(len(data)), {"next_offset": len(data), "events": []})

    def test_partial_final_line_waits_correctly(self) -> None:
        first = self.event("first")
        self.write(first + b'{"type": "sec')
        payload = self.poll(0)
        self.assertEqual(payload, {"next_offset": len(first), "events": ["first"]})
        self.assertEqual(self.poll(len(first)), {"next_offset": len(first), "events": []})
        self.write(b'ond"}\n', "ab")
        self.assertEqual(self.poll(len(first))["events"], ["second"])

    def test_line_longer_than_window_eventually_advances(self) -> None:
        big = json.dumps({"type": "big", "pad": "x" * (300 * 1024)}).encode() + b"\n"
        data = self.event("before") + big + b"not json\n" + self.event("after")
        self.write(data)
        offset, events = self.drain()
        self.assertEqual(offset, len(data))
        self.assertEqual(events, ["before", "big", "after"])

    def test_oversized_line_is_omitted_and_never_repeats_offset(self) -> None:
        oversized = b'{"type": "huge", "pad": "' + b"y" * (api.PROGRESS_MAX_EVENT_BYTES + 4096) + b'"}\n'
        data = self.event("before") + oversized + b"{malformed\n" + self.event("after")
        self.write(data)
        offset, events = self.drain()
        self.assertEqual(offset, len(data))
        self.assertEqual(events, ["before", api.OVERSIZED_EVENT, "after"])

    def test_oversized_line_is_skipped_across_bounded_requests(self) -> None:
        with patch.multiple(
            api,
            PROGRESS_MAX_BYTES=64,
            PROGRESS_MAX_EVENT_BYTES=128,
            PROGRESS_MAX_SKIP_BYTES=100,
            _PROGRESS_SCAN_CHUNK_BYTES=16,
        ):
            data = b"z" * 1000 + b"\n" + b"[1, 2]\n" + b"not json\n" + self.event("after")
            self.write(data)
            offset, events = self.drain()
        self.assertEqual(offset, len(data))
        self.assertEqual(events, [api.OVERSIZED_EVENT, "after"])

    def test_oversized_line_still_being_written_advances(self) -> None:
        with patch.multiple(
            api,
            PROGRESS_MAX_BYTES=64,
            PROGRESS_MAX_EVENT_BYTES=128,
            PROGRESS_MAX_SKIP_BYTES=100,
            _PROGRESS_SCAN_CHUNK_BYTES=16,
        ):
            self.write(b"q" * 500)
            first = self.poll(0)
            self.assertGreater(first["next_offset"], 0)
            self.assertEqual(first["events"], [api.OVERSIZED_EVENT])
            offset, events = self.drain(first["next_offset"])
            self.assertEqual(offset, 500)
            self.assertEqual(events, [])
            self.write(b"q" * 50 + b"\n" + self.event("after"), "ab")
            offset, events = self.drain(offset)
            self.assertEqual(offset, self.path.stat().st_size)
            self.assertEqual(events, ["after"])


if __name__ == "__main__":
    unittest.main()

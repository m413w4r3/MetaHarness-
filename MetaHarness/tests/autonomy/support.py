"""The shared fiction of the autonomy suite.

One property is under test: an operational problem or an ordinary agent error
must lead to autonomous progress of the run, never to a false human stop and
never to an unrecoverable hard stop.  The fakes stay deliberately minimal.

Real: Git, the worktree, the durable run artifacts, the configured checks and
the production state machine.  Fake: every model (scripted planner/reviewer
answers and scripted executors) and every LLM transport, an in-process double
driven by an injected clock.  No socket, no Docker daemon and no provider
endpoint is reachable from this suite.
"""

from __future__ import annotations

import io
import json
import urllib.error
from dataclasses import dataclass
from typing import Any, Mapping
from unittest import mock

from metaharness.llm.chat import OpenAIChatTextClient
from metaharness.models import LLMEndpointConfig, RunStatus
from tests.pipeline_support import PipelineHarness, git

# The fixture SPEC every scenario plans against.  It never names a behaviour
# the assertions search for, so a durable mention of one is never planner text.
SPEC = "Make feature.txt hold the requested content.\n"

# The postures that hand a decision back to an operator.  ``WAIT_HUMAN`` is
# reserved for a real product decision (SPEC_DECISION); a run that lands on one
# of these for an ordinary failure is the false stop the property forbids.
FALSE_HUMAN_STOP_STATUSES = frozenset({
    RunStatus.WAITING_HUMAN,
    RunStatus.WAITING_CHECK_REPAIR,
    RunStatus.WAITING_CONTRACT_REPAIR,
    RunStatus.WAITING_CHECK_INFRASTRUCTURE,
    RunStatus.WAITING_SCOPE_APPROVAL,
})

# The postures of a run that delivered its accepted candidate.
COMPLETED_STATUSES = frozenset({RunStatus.COMMITTED, RunStatus.PUBLISHED})


@dataclass(frozen=True)
class Step:
    """One META PLAN v2 step with its exact declared scope."""

    id: str
    title: str
    read: tuple[str, ...] = ("feature.txt",)
    write: tuple[str, ...] = ()
    create: tuple[str, ...] = ()
    delete: tuple[str, ...] = ()

    def render(self) -> str:
        def block(label: str, values: tuple[str, ...]) -> str:
            lines = "\n".join(f"- {value}" for value in values) if values else "NONE"
            return f"{label}\n{lines}\n"

        reads = tuple(f"{path} :: current content" for path in self.read)
        return f"""BEGIN STEP {self.id}
TITLE: {self.title}
EXECUTION_CLASS: MECHANICAL
DEPENDS_ON: NONE

OBJECTIVE
{self.title}

{block("READ_SET", reads)}
{block("WRITE_SET", self.write)}
{block("CREATE_SET", self.create)}
{block("DELETE_SET", self.delete)}
INSTRUCTIONS
1. {self.title}

VERIFY
- Run the configured checks.

FORBIDDEN
- Do not change paths outside the declared sets.

END STEP {self.id}
"""


def meta_plan(
    *steps: Step,
    required_checks: tuple[str, ...] = ("test",),
    title: str = "Deliver the change",
) -> str:
    """A ready META PLAN v2 whose declared sets are exactly the given ones."""

    mode = "SINGLE" if len(steps) == 1 else "STAGED"
    checks = "\n".join(f"- {check}" for check in required_checks)
    return f"""META PLAN v2

STATUS: READY
TITLE: {title}

OBJECTIVE
Implement the requested change.

CONSTRAINTS
Keep the change local.

EXECUTION_MODE: {mode}
STEP_COUNT: {len(steps)}

{"".join(step.render() for step in steps)}
ACCEPTANCE
The requested change is present.

REQUIRED_CHECKS
{checks}

TESTS
The configured checks are the final evidence.

RISKS
NONE

BLOCKERS
NONE

END META PLAN
"""


def repaired_contract(step_id: str, title: str, path: str) -> str:
    """A valid planner repair of one step contract, keeping its identity."""

    return f"""META STEP CONTRACT REPAIR v1
STEP_ID: {step_id}
TITLE: {title}
EXECUTION_CLASS: MECHANICAL
DEPENDS_ON: NONE

OBJECTIVE
{title} with the corrected bounded scope.

READ_SET
- {path} :: current content

WRITE_SET
- {path}

CREATE_SET
NONE

DELETE_SET
NONE

INSTRUCTIONS
1. Perform the step on {path} only.

VERIFY
- Run the configured checks.

FORBIDDEN
- Do not edit paths outside the approved set.

END META STEP CONTRACT REPAIR
"""


class FakeClock:
    """A monotonic clock whose sleeps advance it: the suite never waits."""

    def __init__(self, start: float = 1_000.0) -> None:
        self.started = start
        self.now = start
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += max(0.0, seconds)

    @property
    def elapsed(self) -> float:
        return self.now - self.started


class _FakeResponse:
    """The minimal response surface ``_read_bounded`` consumes."""

    def __init__(self, body: bytes) -> None:
        self._buffer = io.BytesIO(body)

    def read1(self, size: int = -1) -> bytes:
        return self._buffer.read1(size)

    def read(self, size: int = -1) -> bytes:
        return self._buffer.read(size)

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False


def completion_body(text: str) -> bytes:
    """One valid OpenAI-compatible completion whose whole content is *text*."""

    payload = {"choices": [{"message": {"role": "assistant", "content": text}}]}
    return json.dumps(payload).encode("utf-8")


class FlakyHTTPTransport:
    """An in-process HTTP transport that refuses to answer for a while.

    Every request made while the injected clock has not yet run for
    ``outage_seconds`` fails with an HTTP 503; every later request receives the
    scripted completion.  No socket is ever opened.
    """

    def __init__(self, clock: FakeClock, *, outage_seconds: float, answer: str) -> None:
        self.clock = clock
        self.outage_seconds = outage_seconds
        self.answer = answer
        self.attempts = 0

    def open(self, request: Any, timeout: Any = None) -> _FakeResponse:
        self.attempts += 1
        if self.clock.elapsed < self.outage_seconds:
            raise urllib.error.HTTPError(
                request.full_url, 503, "Service Unavailable", {}, None,
            )
        return _FakeResponse(completion_body(self.answer))


def chat_endpoint() -> LLMEndpointConfig:
    """The endpoint the fake transport answers for; it is never dialed.

    It carries the default transport horizon, exactly as the live profiles
    do: surviving an outage can only come from the time-based horizon.
    """

    return LLMEndpointConfig(
        base_url="http://127.0.0.1:9",
        endpoint_path="/v1/chat/completions",
        model="scripted",
        timeout_seconds=30,
    )


class CompletionOverTransport:
    """A chat client whose only transport is the injected fake opener.

    ``time`` is replaced by the injected clock for the duration of one request,
    so a retry policy that waits for an outage to end costs no wall time.
    """

    def __init__(
        self,
        endpoint: LLMEndpointConfig,
        transport: FlakyHTTPTransport,
        clock: FakeClock,
    ) -> None:
        self._clock = clock
        self._client = OpenAIChatTextClient(endpoint, environment={}, opener=transport)

    def complete(self, request: str) -> Any:
        with (
            mock.patch("time.monotonic", self._clock.monotonic),
            mock.patch("time.sleep", self._clock.sleep),
        ):
            return self._client.complete(request)


class AutonomyHarness(PipelineHarness):
    """A real repository and a real run, driven by scripted models only."""

    def commit_files(self, files: Mapping[str, str]) -> None:
        """Write *files* into the fixture repository and publish them on main."""

        for relative, content in files.items():
            path = self.repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        git(self.repo, "add", "--all")
        git(self.repo, "commit", "-qm", "fixture: " + ", ".join(sorted(files)))
        git(self.repo, "push", "-q", "origin", "main")

    def green_check(self) -> None:
        """Replace the fixture check with one that always passes."""

        self.check.write_text("import sys\nsys.exit(0)\n", encoding="utf-8")

    def failure_reason(self) -> str | None:
        """The stable failure code the run recorded, if it ended on one."""

        failure = self.state().get("failure") or {}
        return failure.get("reason") if isinstance(failure, dict) else None

    def step_record(self, step_id: str = "S01", *, cycle: int = 1) -> dict[str, Any]:
        """The durable record of one implementation step."""

        path = self.run_dir() / f"cycles/{cycle:03d}/implementation/steps/{step_id}/step.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def durable_report(self, *, run_id: str = "run", limit: int = 256 * 1024) -> str:
        """A bounded view of the run's durable decision artifacts.

        Normalizations, warnings and baseline verdicts have no frozen module to
        live in yet, so the scenarios read them where the run keeps its record.
        """

        view: list[str] = []
        total = 0
        for path in sorted(self.run_dir(run_id).rglob("*")):
            if not path.is_file() or path.suffix not in {".json", ".md"}:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")[:32 * 1024]
            view.append(text)
            total += len(text)
            if total >= limit:
                break
        return "\n".join(view)

    def assert_not_false_human_stop(self, result: Any) -> None:
        """The run must not wait for an operator decision it did not need."""

        self.assertNotIn(
            result.status, FALSE_HUMAN_STOP_STATUSES,
            f"false human stop: status={result.status} reason={self.failure_reason()}",
        )

    def assert_not_unrecoverable_hard_stop(self, result: Any) -> None:
        """A recoverable problem must never end the run as an unrecoverable failure."""

        self.assertNotEqual(
            result.status, RunStatus.FAILED,
            f"unrecoverable hard stop: reason={self.failure_reason()}",
        )

    def assert_run_completed(self, result: Any) -> None:
        """The run delivered its candidate instead of stopping short."""

        self.assertIn(
            result.status, COMPLETED_STATUSES,
            f"run did not complete: status={result.status} reason={self.failure_reason()}",
        )

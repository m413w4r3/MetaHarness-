"""Background execution manager for runs submitted through the local UI."""

from __future__ import annotations

import threading
from typing import Callable

from ..config import HarnessConfig
from ..orchestrator import Orchestrator, OrchestrationError, _safe_run_id, generate_run_id
from ..plan_recovery import PlanRecoveryError
from ..resume import ResumeNotAllowedError
from ..run_options import RunOptions


class RunManagerError(RuntimeError):
    pass


class RunCapacityError(RunManagerError):
    pass


class RunCollisionError(RunManagerError):
    pass


_CREATION_TIMEOUT_SECONDS = 5.0


class RunManager:
    def __init__(
        self,
        config: HarnessConfig,
        *,
        max_active_runs: int = 1,
        orchestrator_factory: Callable[[HarnessConfig], Orchestrator] | None = None,
    ) -> None:
        if max_active_runs < 1:
            raise ValueError("max_active_runs must be positive")
        self._config = config
        self._max_active_runs = max_active_runs
        self._orchestrator_factory = orchestrator_factory or Orchestrator
        self._lock = threading.Lock()
        self._active_run_ids: set[str] = set()

    def start_run(
        self,
        spec: str,
        *,
        run_id: str | None = None,
        planner_profile: str | None = None,
        run_options: RunOptions | None = None,
    ) -> str:
        try:
            selected_run_id = _safe_run_id(run_id) if run_id else generate_run_id()
        except (OrchestrationError, TypeError) as exc:
            raise RunManagerError(str(exc)) from exc

        with self._lock:
            if selected_run_id in self._active_run_ids:
                raise RunCollisionError("run already exists or is active")
            if len(self._active_run_ids) >= self._max_active_runs:
                raise RunCapacityError("maximum active runs reached")
            self._active_run_ids.add(selected_run_id)

        created_event = threading.Event()
        finished_event = threading.Event()
        signal = threading.Condition()

        def notify(event: threading.Event) -> None:
            with signal:
                event.set()
                signal.notify_all()

        def worker() -> None:
            try:
                orchestrator = self._orchestrator_factory(self._config)
                kwargs = {
                    "run_id": selected_run_id,
                    "on_created": lambda _run_dir: notify(created_event),
                }
                if planner_profile is not None:
                    kwargs["planner_profile"] = planner_profile
                if run_options is not None:
                    kwargs["run_options"] = run_options
                orchestrator.run_text(spec, **kwargs)
            except BaseException:
                # The orchestrator records business failures after durable
                # initialization. Factory/thread failures must not escape the
                # daemon thread or strand capacity forever.
                pass
            finally:
                # Capacity is released before the caller is woken, so a
                # failure reported by start_run() has already restored it.
                with self._lock:
                    self._active_run_ids.discard(selected_run_id)
                notify(finished_event)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        with signal:
            signal.wait_for(
                lambda: created_event.is_set() or finished_event.is_set(),
                timeout=_CREATION_TIMEOUT_SECONDS,
            )
        if created_event.is_set():
            return selected_run_id
        if finished_event.is_set():
            raise RunManagerError("run worker failed before durable creation")
        raise RunManagerError("run creation timed out")


    def resume_run(self, run_id: str) -> str:
        """Resume the same ``run_id`` in the background at its checkpoint."""

        return self._run_until_claimed(
            run_id,
            lambda orchestrator, run, on_claimed: orchestrator.resume(run, on_claimed=on_claimed),
            failure="run could not be resumed",
        )

    def recover_plan(self, run_id: str, replacement_raw: str) -> str:
        """Publish an operator plan for ``run_id``, then resume it to approval.

        Validation, persistence and the resume claim happen in the worker
        before this returns; no planner is ever called.
        """

        return self._run_until_claimed(
            run_id,
            lambda orchestrator, run, on_claimed: orchestrator.recover_plan(
                run, replacement_raw, on_claimed=on_claimed
            ),
            failure="plan could not be recovered",
        )

    def _run_until_claimed(
        self,
        run_id: str,
        action: Callable[[Orchestrator, str, Callable[[object], None]], object],
        *,
        failure: str,
    ) -> str:
        try:
            selected_run_id = _safe_run_id(run_id)
        except (OrchestrationError, TypeError) as exc:
            raise RunManagerError(str(exc)) from exc
        with self._lock:
            if selected_run_id in self._active_run_ids:
                raise RunCollisionError("run already exists or is active")
            if len(self._active_run_ids) >= self._max_active_runs:
                raise RunCapacityError("maximum active runs reached")
            self._active_run_ids.add(selected_run_id)

        claimed_event = threading.Event()
        finished_event = threading.Event()
        signal = threading.Condition()
        errors: list[BaseException] = []

        def notify(event: threading.Event) -> None:
            with signal:
                event.set()
                signal.notify_all()

        def worker() -> None:
            try:
                orchestrator = self._orchestrator_factory(self._config)
                action(orchestrator, selected_run_id, lambda _run_dir: notify(claimed_event))
            except BaseException as exc:  # recorded, never escapes the thread
                errors.append(exc)
            finally:
                with self._lock:
                    self._active_run_ids.discard(selected_run_id)
                notify(finished_event)

        threading.Thread(target=worker, daemon=True).start()
        with signal:
            signal.wait_for(
                lambda: claimed_event.is_set() or finished_event.is_set(),
                timeout=_RESUME_VALIDATION_TIMEOUT_SECONDS,
            )
        if not claimed_event.is_set() and finished_event.is_set() and errors:
            if isinstance(errors[0], PlanRecoveryError):
                raise RunPlanRecoveryError(str(errors[0]))
            if isinstance(errors[0], ResumeNotAllowedError):
                raise RunResumeNotAllowedError(str(errors[0]))
            raise RunManagerError(failure)
        return selected_run_id


class RunResumeNotAllowedError(RunManagerError):
    pass


class RunPlanRecoveryError(RunManagerError):
    """The operator plan or the run was refused; the run is unchanged."""


# Resume validation reads Git trees (no model call); allow it more time than
# a creation before answering the browser, which then shows durable state.
_RESUME_VALIDATION_TIMEOUT_SECONDS = 60.0


__all__ = [
    "RunCapacityError",
    "RunCollisionError",
    "RunManager",
    "RunManagerError",
    "RunPlanRecoveryError",
    "RunResumeNotAllowedError",
]
